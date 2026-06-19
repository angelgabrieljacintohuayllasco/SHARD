"""
IVF-PQ self-check: recall@10 vs exact brute force on synthetic clustered data,
plus disk-residency assertions. No real dataset needed.

Run standalone:   python -m tests.test_ivfpq      (from SHARD-main/)
Or via pytest:     pytest tests/test_ivfpq.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np

from shard.index.ivfpq_builder import build_ivfpq
from shard.index.ivfpq_reader import IVFPQReader


def make_clustered(n, dim=384, n_centers=512, spread=0.10, seed=0):
    """Clustered gaussians, L2-normalized — mimics real embedding geometry so
    that approximate NN is meaningful (uniform random would be equidistant)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_centers, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    lbl = rng.integers(0, n_centers, n)
    X = centers[lbl] + spread * rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    keys = [f"doc_{i}" for i in range(n)]
    return X.astype(np.float32), keys


def brute_topk(X, q, k):
    s = X @ q
    idx = np.argpartition(-s, k)[:k]
    return set(int(i) for i in idx)


def _run(profile, n=100_000, nprobe=None, n_queries=200, gate=0.85, seed=1):
    X, keys = make_clustered(n, seed=seed)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        out = Path(td) / "ivf"
        build_ivfpq(X, keys, str(out), profile=profile, seed=seed)
        rdr = IVFPQReader(str(out))
        try:
            # disk-residency gates
            assert isinstance(rdr.codes_mm, np.memmap), "codes must be a memmap (disk-resident)"
            assert rdr.centroids.size < n * rdr.dim, "no full (N,dim) array may be RAM-resident"

            rng = np.random.default_rng(seed + 1)
            qidx = rng.integers(0, n, n_queries)
            rdr.search(X[qidx[0]], 10, nprobe=nprobe)          # warmup (page cache)

            recs, lats = [], []
            for qi in qidx:
                q = X[qi]
                gt = brute_topk(X, q, 10)
                t0 = time.perf_counter()
                ann = rdr.search(q, 10, nprobe=nprobe)
                lats.append(time.perf_counter() - t0)
                got = set(int(k.split("_")[1]) for k, _ in ann)
                recs.append(len(gt & got) / 10)

            R = float(np.mean(recs))
            p50 = float(np.percentile(lats, 50) * 1e3)
            p95 = float(np.percentile(lats, 95) * 1e3)
            print(f"[{profile}] recall@10={R:.3f}  p50={p50:.2f}ms p95={p95:.2f}ms  "
                  f"N={n} K={rdr.K} m={rdr.m} nprobe={nprobe or rdr.nprobe_default} rerank={rdr.rerank}")
            assert R >= gate, f"recall regression {R:.3f} < {gate}"
            return R
        finally:
            rdr.close()


def test_recall_low_ram():
    _run("low-ram", gate=0.85)


def test_recall_medium():
    _run("medium", gate=0.88)


def test_recall_fast():
    _run("fast", gate=0.92)


if __name__ == "__main__":
    _run("low-ram", gate=0.85)
    _run("medium", gate=0.88)
    _run("fast", gate=0.92)
    print("OK")
