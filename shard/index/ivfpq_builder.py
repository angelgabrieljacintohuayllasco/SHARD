"""
IVF-PQ Builder — streaming, RAM-bounded index build (numpy + sklearn at build only).

Builds an IVF (coarse k-means) + PQ-on-residual index that scales toward ~1B
vectors while keeping build RAM bounded to roughly: training sample + one chunk
+ coarse centroids. The full N x dim matrix is never materialized in RAM.

Designed to run OFFLINE on a powerful machine (Colab GPU / strong PC). The
resulting read-only artifact is copied to a low-RAM device for query.

CLI:
  python -m shard.index.ivfpq_builder --embeddings emb.npy --keys keys.json \
      --out ./mydb/ivf --profile low-ram
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from shard.index.ivfpq_format import (
    KSUB, PROFILES, FILES, choose_K, default_nprobe, default_rerank, write_manifest,
)

# Tile size for the (rows x K) score matrix in assignment, ~256 MB of float32.
_ASSIGN_TILE_ENTRIES = 64_000_000


def _assign_to_centroids(X, centroids, cnorm, out):
    """argmin_c ||x - c||^2 for each row of X, tiled over rows to bound memory.

    ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 is constant per row so we
    minimize (||c||^2 - 2 x.c). Writes cluster ids into `out` (len == len(X)).
    """
    n = len(X)
    tile = max(1, _ASSIGN_TILE_ENTRIES // max(1, centroids.shape[0]))
    for s in range(0, n, tile):
        e = min(n, s + tile)
        scores = X[s:e].astype(np.float32) @ centroids.T   # (b, K)
        scores *= -2.0
        scores += cnorm
        out[s:e] = scores.argmin(1)


def _pq_encode(resid, codebooks, m, dsub):
    """Encode residuals (B, dim) -> PQ codes (B, m) uint8."""
    b = len(resid)
    codes = np.empty((b, m), np.uint8)
    for j in range(m):
        sub = resid[:, j * dsub:(j + 1) * dsub]            # (B, dsub)
        cb = codebooks[j]                                  # (256, dsub)
        d = (sub * sub).sum(1)[:, None] - 2.0 * (sub @ cb.T) + (cb * cb).sum(1)
        codes[:, j] = d.argmin(1).astype(np.uint8)
    return codes


def build_ivfpq(vectors, keys, out_dir, profile="low-ram", seed=0, rerank="auto"):
    """Build an IVF-PQ index.

    Args:
        vectors: (N, dim) ndarray OR np.memmap (e.g. np.load(path, mmap_mode='r')).
                 Rows must be L2-normalized. Read in chunks; never fully loaded.
        keys:    sequence of N keys (str). keys[i] identifies the SHARD record
                 for vector row i (used later by MMapReader.find).
        out_dir: output directory for the ivf/ artifact.
        profile: "low-ram" | "medium" | "fast".
        rerank:  rerank cache mode "auto" | "none" | "sq8" | "f32". "auto" picks
                 per profile (sq8 for low-ram/medium, f32 for fast). PQ alone
                 ranks near-ties poorly; the cache re-scores the shortlist with
                 (near-)exact dot. It lives on disk (mmap) — only the shortlist
                 rows are paged into RAM, so even sq8 keeps the low-RAM promise.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")
    if rerank == "auto":
        rerank = default_rerank(profile)
    if rerank not in ("none", "sq8", "f32"):
        raise ValueError(f"unknown rerank {rerank!r}")

    N, dim = int(vectors.shape[0]), int(vectors.shape[1])
    if N == 0:
        raise ValueError("no vectors to index")
    if len(keys) != N:
        raise ValueError(f"keys ({len(keys)}) and vectors ({N}) length mismatch")

    p = PROFILES[profile]
    m, chunk = p["m"], p["chunk"]
    if dim % m != 0:
        raise ValueError(f"dim {dim} not divisible by m {m} for profile {profile}")
    dsub = dim // m

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ── 1. Training sample (random rows; only the sample is read into RAM) ─────
    n_sample = min(N, p["sample"])
    sidx = np.sort(rng.choice(N, size=n_sample, replace=False))
    sample = np.asarray(vectors[sidx], dtype=np.float32)

    # ── 2. Coarse quantizer (IVF centroids) ───────────────────────────────────
    K = min(choose_K(N), n_sample)
    print(f"[IVF-PQ] N={N:,} dim={dim} profile={profile} K={K} m={m} dsub={dsub}")
    print(f"[IVF-PQ] training coarse k-means on {n_sample:,} sample vectors...")
    mbk = MiniBatchKMeans(n_clusters=K, batch_size=10_000, n_init=3,
                          max_iter=100, random_state=seed)
    mbk.fit(sample)
    centroids = mbk.cluster_centers_.astype(np.float32)
    cnorm = (centroids * centroids).sum(1).astype(np.float32)

    # ── 3. PQ codebooks on residuals of the sample ────────────────────────────
    print("[IVF-PQ] training PQ codebooks on residuals...")
    samp_assign = np.empty(n_sample, np.int64)
    _assign_to_centroids(sample, centroids, cnorm, samp_assign)
    resid = sample - centroids[samp_assign]
    codebooks = np.zeros((m, KSUB, dsub), np.float32)
    for j in range(m):
        sub = resid[:, j * dsub:(j + 1) * dsub]
        kk = min(KSUB, len(np.unique(sub, axis=0)))
        kk = max(1, kk)
        km = MiniBatchKMeans(n_clusters=kk, batch_size=10_000, n_init=3,
                             random_state=seed).fit(sub)
        codebooks[j, :kk] = km.cluster_centers_.astype(np.float32)
    del sample, resid, samp_assign
    gc.collect()

    # ── 4. Pass A: assign every vector to a list; tally per-list counts ────────
    print("[IVF-PQ] pass A: assigning vectors to lists...")
    assign_path = out / "_assign.tmp.i64"
    assign_mm = np.memmap(assign_path, dtype=np.int64, mode="w+", shape=(N,))
    counts = np.zeros(K, np.int64)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Xc = np.asarray(vectors[s:e], dtype=np.float32)
        a = np.empty(e - s, np.int64)
        _assign_to_centroids(Xc, centroids, cnorm, a)
        assign_mm[s:e] = a
        counts += np.bincount(a, minlength=K)
    assign_mm.flush()
    offsets = np.zeros(K + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])

    # ── 5. Pass B: encode residuals + scatter into contiguous list-major files ─
    print("[IVF-PQ] pass B: encoding + scattering codes...")
    codes_mm = np.memmap(out / FILES["list_codes"], dtype=np.uint8, mode="w+", shape=(N, m))
    row_to_orig = np.memmap(out / FILES["row_to_orig"], dtype=np.int64, mode="w+", shape=(N,))
    if rerank == "f32":
        rerank_mm = np.memmap(out / FILES["rerank_f32"], dtype=np.float32, mode="w+", shape=(N, dim))
    elif rerank == "sq8":
        rerank_mm = np.memmap(out / FILES["rerank_sq8"], dtype=np.int8, mode="w+", shape=(N, dim))
    else:
        rerank_mm = None
    cursor = offsets[:-1].copy()                            # next free row per list
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        b = e - s
        Xc = np.asarray(vectors[s:e], dtype=np.float32)
        a = np.asarray(assign_mm[s:e])
        resid = Xc - centroids[a]
        codes = _pq_encode(resid, codebooks, m, dsub)
        # vectorized scatter: group this chunk's rows by cluster, place each
        # group into a contiguous span starting at the cluster's cursor.
        order = np.argsort(a, kind="stable")
        uc, first, cnt = np.unique(a[order], return_index=True, return_counts=True)
        dest = np.empty(b, np.int64)
        for ui in range(len(uc)):
            c = int(uc[ui])
            base = int(cursor[c])
            k = int(cnt[ui])
            rows = order[first[ui]:first[ui] + k]
            dest[rows] = base + np.arange(k)
            cursor[c] = base + k
        codes_mm[dest] = codes
        row_to_orig[dest] = np.arange(s, e)
        if rerank == "f32":
            rerank_mm[dest] = Xc
        elif rerank == "sq8":
            # L2-normalized components are in ~[-1,1]; int8 = round(x*127)
            rerank_mm[dest] = np.clip(np.rint(Xc * 127.0), -127, 127).astype(np.int8)
    codes_mm.flush()
    row_to_orig.flush()
    if rerank_mm is not None:
        rerank_mm.flush()

    # ── 6. Keys (original order) + byte offsets ────────────────────────────────
    # ponytail: O(N) python loop, build-time only (offline). For >~100M keys,
    # stream keys from disk instead of holding the list in RAM.
    key_off = np.zeros(N + 1, np.int64)
    with open(out / FILES["keys"], "wb") as f:
        cur = 0
        for i in range(N):
            kb = str(keys[i]).encode("utf-8")
            f.write(kb)
            cur += len(kb)
            key_off[i + 1] = cur
    key_off.astype("<i8").tofile(out / FILES["key_offsets"])

    # ── 7. Persist RAM-resident arrays (raw little-endian) + manifest ──────────
    centroids.astype("<f4").tofile(out / FILES["coarse_centroids"])
    codebooks.astype("<f4").tofile(out / FILES["pq_codebooks"])
    offsets.astype("<i8").tofile(out / FILES["list_offsets"])

    manifest = {
        "version": 1,
        "profile": profile,
        "dim": dim,
        "metric": "ip_via_l2_residual",
        "K": int(K),
        "m": int(m),
        "dsub": int(dsub),
        "ksub": KSUB,
        "use_residual": True,
        "n_total": int(N),
        "nprobe_default": int(default_nprobe(profile, K)),
        "rerank": rerank,
        "files": FILES,
    }
    write_manifest(out, manifest)

    # ── 8. Drop temp assign file ───────────────────────────────────────────────
    del assign_mm, codes_mm, row_to_orig, rerank_mm
    gc.collect()
    try:
        assign_path.unlink()
    except OSError:
        pass

    print(f"[IVF-PQ] done. index at {out}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an IVF-PQ index for SHARD")
    ap.add_argument("--embeddings", required=True, help="embeddings.npy, shape (N,dim), L2-normalized")
    ap.add_argument("--keys", required=True, help="JSON list of N keys (or {idx:key} map)")
    ap.add_argument("--out", required=True, help="output ivf/ directory")
    ap.add_argument("--profile", default="low-ram", choices=list(PROFILES))
    ap.add_argument("--rerank", default="auto", choices=["auto", "none", "sq8", "f32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vectors = np.load(args.embeddings, mmap_mode="r")       # never fully loaded
    with open(args.keys, encoding="utf-8") as f:
        raw = json.load(f)
    keys = raw if isinstance(raw, list) else [raw[str(i)] for i in range(len(raw))]
    build_ivfpq(vectors, keys, args.out, profile=args.profile, rerank=args.rerank, seed=args.seed)


if __name__ == "__main__":
    main()
