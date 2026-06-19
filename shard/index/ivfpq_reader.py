"""
IVF-PQ Reader — query-time search. Pure numpy, disk-resident via mmap.

RAM-resident after construction:
  - coarse centroids   (K, dim) float32
  - PQ codebooks       (m, 256, dsub) float32   (< 0.4 MB)
  - list offsets       (K+1,) int64
Everything else (PQ codes, row->orig map, key offsets, keys, optional rerank
vectors) is memory-mapped read-only and only the probed slices are paged in.
This is what lets a 2 GB box query a 1 TB index.

Search uses residual PQ with asymmetric distance computation (ADC): for unit
vectors, argmax dot == argmin ||q - x||^2, and PQ approximates that L2 cleanly.
"""

from pathlib import Path

import numpy as np

from shard.index.ivfpq_format import read_manifest


class IVFPQReader:
    def __init__(self, ivf_dir):
        self.dir = Path(ivf_dir)
        m = read_manifest(ivf_dir)
        self.manifest = m
        self.dim = int(m["dim"])
        self.m = int(m["m"])
        self.dsub = int(m["dsub"])
        self.K = int(m["K"])
        self.ksub = int(m["ksub"])
        self.n_total = int(m["n_total"])
        self.nprobe_default = int(m["nprobe_default"])
        F = m["files"]

        # RAM-resident (small)
        self.centroids = np.fromfile(self.dir / F["coarse_centroids"], dtype="<f4").reshape(self.K, self.dim)
        self._cnorm = (self.centroids * self.centroids).sum(1).astype(np.float32)
        self.codebooks = np.fromfile(self.dir / F["pq_codebooks"], dtype="<f4").reshape(self.m, self.ksub, self.dsub)
        self._cb_sqnorm = (self.codebooks * self.codebooks).sum(2).astype(np.float32)  # (m, 256)
        self.offsets = np.fromfile(self.dir / F["list_offsets"], dtype="<i8")

        # Disk-resident (mmap). Validate size so a bad/partial copy fails loud,
        # and the reshape is guaranteed correct (fixes the old flat-memmap bug).
        codes_path = self.dir / F["list_codes"]
        expect = self.n_total * self.m
        actual = codes_path.stat().st_size
        if actual != expect:
            raise ValueError(f"list_codes size {actual} != expected {expect} (n_total*m)")
        self.codes_mm = np.memmap(codes_path, dtype=np.uint8, mode="r").reshape(self.n_total, self.m)
        self.row_to_orig = np.memmap(self.dir / F["row_to_orig"], dtype="<i8", mode="r")
        self.key_offsets = np.memmap(self.dir / F["key_offsets"], dtype="<i8", mode="r")
        self._keys_mm = np.memmap(self.dir / F["keys"], dtype=np.uint8, mode="r")

        # Rerank cache (disk-resident; only the shortlist rows are paged in)
        self.rerank = m.get("rerank", "none")
        if self.rerank == "f32":
            self._rerank_mm = np.memmap(self.dir / F["rerank_f32"], dtype="<f4", mode="r").reshape(self.n_total, self.dim)
        elif self.rerank == "sq8":
            self._rerank_mm = np.memmap(self.dir / F["rerank_sq8"], dtype=np.int8, mode="r").reshape(self.n_total, self.dim)
        else:
            self._rerank_mm = None
        self.has_rerank = self._rerank_mm is not None
        self._marange = np.arange(self.m)

    def _key(self, g: int) -> str:
        orig = int(self.row_to_orig[g])
        a = int(self.key_offsets[orig])
        b = int(self.key_offsets[orig + 1])
        return bytes(self._keys_mm[a:b]).decode("utf-8")

    def search(self, q, top_k=10, nprobe=None, overfetch=8, rerank=None, rerank_pool=256):
        """Return [(key, score)] for the top_k nearest.

        ADC builds a candidate shortlist from the probed lists; if a rerank
        cache exists it re-scores the shortlist with (near-)exact dot. score is
        the exact/sq8 dot when reranked, else the approx dot from ADC.
        """
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
        nprobe = int(nprobe or self.nprobe_default)
        nprobe = max(1, min(nprobe, self.K))

        # coarse: pick nprobe nearest lists (argmin L2 to centroid)
        cdist = self._cnorm - 2.0 * (self.centroids @ q)        # (K,)
        if nprobe < self.K:
            probe = np.argpartition(cdist, nprobe - 1)[:nprobe]
        else:
            probe = np.arange(self.K)

        rows_all, approx_all = [], []
        for c in probe:
            s = int(self.offsets[c]); e = int(self.offsets[c + 1])
            if e <= s:
                continue
            r_q = (q - self.centroids[c]).reshape(self.m, self.dsub)        # (m, dsub)
            # ADC table: lut[j, code] = ||r_q_j - codebook[j, code]||^2
            lut = self._cb_sqnorm - 2.0 * np.einsum("mkd,md->mk", self.codebooks, r_q)
            lut += (r_q * r_q).sum(1)[:, None]
            codes_c = np.asarray(self.codes_mm[s:e])           # (L, m) uint8
            approx = lut[self._marange, codes_c].sum(1)        # (L,)
            rows_all.append(np.arange(s, e))
            approx_all.append(approx)

        if not rows_all:
            return []

        rows = np.concatenate(rows_all)
        approx = np.concatenate(approx_all)

        do_rerank = self.has_rerank if rerank is None else (rerank and self._rerank_mm is not None)
        # Shortlist size: a generous pool when reranking (exact re-score is cheap
        # and a bigger pool catches near-ties PQ ranks poorly), else just top_k*overfetch.
        pool = max(top_k * max(1, overfetch), rerank_pool) if do_rerank else top_k * max(1, overfetch)
        pool = min(pool, len(rows))
        if pool < len(rows):
            sel = np.argpartition(approx, pool - 1)[:pool]
        else:
            sel = np.arange(len(rows))

        g_rows = rows[sel]
        if do_rerank:
            vecs = np.asarray(self._rerank_mm[g_rows]).astype(np.float32)   # (pool, dim)
            if self.rerank == "sq8":
                vecs *= (1.0 / 127.0)
            dots = vecs @ q
            order = np.argsort(-dots)[:top_k]
            return [(self._key(int(g)), float(d)) for g, d in zip(g_rows[order], dots[order])]

        # ADC-only: smaller approx L2 = closer; dot = 1 - 0.5*||q-x||^2 for unit vecs
        g_approx = approx[sel]
        order = np.argsort(g_approx)[:top_k]
        return [(self._key(int(g_rows[i])), float(1.0 - 0.5 * g_approx[i])) for i in order]

    def close(self) -> None:
        """Release mmap handles (needed on Windows before deleting the dir)."""
        for a in (self.codes_mm, self.row_to_orig, self.key_offsets, self._keys_mm, self._rerank_mm):
            mm = getattr(a, "_mmap", None)
            if mm is not None:
                mm.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
