"""
IVF-PQ index format — profiles, parameter rules, and manifest I/O.

The on-disk index is raw little-endian C-contiguous binary (no .npy headers)
plus a JSON manifest that records every shape/dtype/param. The manifest is the
single source of truth for reshaping mmap'd arrays — this is what makes the
reader's reshape unambiguous and portable across machines (Linux build box ->
Windows laptop). Only relative filenames are stored, never absolute paths.
"""

import json
import math
from pathlib import Path

KSUB = 256          # PQ sub-codebook size (8-bit codes)
IVF_DIRNAME = "ivf"

# m = PQ subquantizers. 384 dims must be divisible by m -> dsub = 384/m.
#   low-ram : finest codes (best recall), bigger on disk but mmap'd (RAM stays low)
#   fast    : smallest codes, pair with full-vector rerank when N is small
PROFILES = {
    "low-ram": {"m": 48, "sample": 256_000,   "chunk": 50_000},
    "medium":  {"m": 32, "sample": 1_000_000, "chunk": 100_000},
    "fast":    {"m": 16, "sample": 2_000_000, "chunk": 100_000},
}

FILES = {
    "coarse_centroids": "coarse_centroids.f32",   # <f4 [K, dim]
    "pq_codebooks":     "pq_codebooks.f32",       # <f4 [m, 256, dsub]
    "list_codes":       "list_codes.u8",          # |u1 [n_total, m]  list-major
    "list_offsets":     "list_offsets.i64",       # <i8 [K+1]
    "row_to_orig":      "row_to_orig.i64",         # <i8 [n_total]  list-major row -> original index
    "keys":             "keys.bin",                # concatenated UTF-8 keys, original order
    "key_offsets":      "key_offsets.i64",         # <i8 [n_total+1]  original-indexed byte offsets
    "rerank_f32":       "rerank_vecs.f32",         # <f4 [n_total, dim]  exact rerank cache
    "rerank_sq8":       "rerank_vecs.i8",          # |i1 [n_total, dim]  int8 rerank cache (/127)
}

# Rerank cache: how shortlisted candidates are re-scored with (near-)exact dot.
#   "none" : ADC score only (lowest recall, smallest artifact)
#   "sq8"  : int8 reconstruction, 384 B/vec, mmap'd — high recall at low RAM
#   "f32"  : exact float32, 1536 B/vec — highest recall, biggest artifact
def default_rerank(profile: str) -> str:
    return {"low-ram": "sq8", "medium": "sq8", "fast": "f32"}[profile]


def choose_K(n: int) -> int:
    """Coarse-cluster count. ~4*sqrt(N), clamped, never more than N points."""
    k = round(4 * math.sqrt(max(1, n)))
    k = max(1024, min(262_144, k))
    return max(1, min(k, n))


def default_nprobe(profile: str, K: int) -> int:
    """Lists probed per query. More = better recall, slower. Capped per profile."""
    frac = {"low-ram": 0.02, "medium": 0.04, "fast": 0.08}[profile]
    cap = {"low-ram": 32, "medium": 64, "fast": 128}[profile]
    return int(max(8, min(cap, max(1, round(K * frac)), K)))


def write_manifest(ivf_dir, manifest: dict) -> None:
    with open(Path(ivf_dir) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(ivf_dir) -> dict:
    with open(Path(ivf_dir) / "manifest.json", encoding="utf-8") as f:
        return json.load(f)
