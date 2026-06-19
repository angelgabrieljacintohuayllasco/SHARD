"""
DEPRECATED — superseded by shard.index.ivfpq_reader (IVFPQReader).
This prototype had broken mmap reshape (read the .npy header as data, never
reshaped to (n,dim)), wrong int8 dequant, and nprobe=1. Use IVFPQReader instead.

IVF Reader — Lazy IVF para query-time en low-RAM.

Carga:
  - centroids.npy FULL en RAM (~6MB)
  - cluster_NNNN.npy como memmap (solo pages needed)

Uso interno desde DASA retrieval_agent.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

class IVFReader:
    def __init__(self, ivf_dir: str):
        self.ivf_dir = Path(ivf_dir)
        self.centroids = np.load(self.ivf_dir / "centroids.npy")  # (K, 384)
        self.dims = self.centroids.shape[1]
        self._cluster_files = list(self.ivf_dir.glob("cluster_*.npy"))
        self._cluster_map = {int(p.stem.split("_")[-1]): p for p in self._cluster_files}
        print(f"[IVFReader] {len(self.centroids)} centroids cargados ({self.centroids.nbytes/1e6:.1f}MB)")

    def find_nearest_cluster(self, query_emb: np.ndarray) -> int:
        """Cosine vs todos centroids → nearest cluster ID. O(K * dim) ~1ms."""
        # Asumir L2-normalized
        scores = self.centroids @ query_emb  # dot = cosine si normalized
        return np.argmax(scores)

    def get_cluster(self, cid: int, top_k: int = 100) -> Tuple[np.ndarray, List[str]]:
        """Lazy load cluster embeddings (int8) + keys."""
        if cid not in self._cluster_map:
            return np.array([]), []
        
        emb_path = self._cluster_map[cid]
        keys_path = emb_path.with_suffix(".keys")
        
        # Memmap para no cargar full en RAM
        cluster_emb = np.memmap(str(emb_path), dtype=np.int8, mode="r")
        
        with open(keys_path) as f:
            cluster_keys = json.load(f)
        
        # Dequantize top_k
        if len(cluster_emb) > top_k:
            # Simple: top por orden archivo (mejorar con scores internos)
            cluster_emb = cluster_emb[:top_k].copy()
            cluster_keys = cluster_keys[:top_k]
        
        # Dequantize int8 -> float32
        cluster_emb = cluster_emb.astype(np.float32) / 127.0  # approx inverse
        
        return cluster_emb, cluster_keys

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Full IVF search: centroids → 1 cluster → cosine top-k."""
        cid = self.find_nearest_cluster(query_emb)
        cluster_emb, keys = self.get_cluster(cid, top_k * 10)
        
        if len(cluster_emb) == 0:
            return []
        
        scores = cluster_emb @ query_emb[:len(cluster_emb)]  # cosine
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        return [(keys[i], float(scores[i])) for i in top_idx]
