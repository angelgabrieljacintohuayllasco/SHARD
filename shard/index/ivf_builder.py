"""
DEPRECATED — superseded by shard.index.ivfpq_builder (build_ivfpq).
This prototype loaded all embeddings into RAM (np.load) and produced an index
the old ivf_reader could not read correctly. Kept only for reference; do not use.

IVF Builder — Offline Inverted File Index para embeddings en TB scale.

Uso:
  python -m shard.index.ivf_builder --embeddings embeddings.npy --keys embedding_keys.json --out ivf/ --k 4096

Genera:
  ivf/centroids.npy     (K x 384 float32, ~6MB)
  ivf/cluster_NNNN.npy  (embeddings int8 del cluster)
  ivf/cluster_NNNN.keys (keys del cluster)
"""

import argparse
import json
import struct
from pathlib import Path
from typing import List

import numpy as np
from sklearn.cluster import KMeans

def quantize_int8(emb: np.ndarray) -> np.ndarray:
    """Cuantiza float32 [-1,1] -> int8 [-127,127] con rangos por cluster."""
    minv, maxv = emb.min(), emb.max()
    scale = 254 / (maxv - minv + 1e-8)
    return np.clip((emb - minv) * scale - 127, -127, 127).astype(np.int8)

def build_ivf(embed_path: str, keys_path: str, out_dir: str, k: int = 4096):
    print(f"[IVF] Cargando {embed_path}...")
    embeddings = np.load(embed_path)  # (N, 384)
    with open(keys_path) as f:
        keys = json.load(f)

    assert len(keys) == len(embeddings), "Keys y embeddings deben coincidir"

    print(f"[IVF] K-means K={k} sobre {len(embeddings):,} embeddings...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    np.save(f"{out_dir}/centroids.npy", centroids)
    print(f"  Centroides guardados: {centroids.nbytes / 1e6:.1f} MB")

    # Agrupar por cluster
    clusters = [[] for _ in range(k)]
    for i, cid in enumerate(cluster_ids):
        clusters[cid].append(i)

    for cid in range(k):
        if not clusters[cid]:
            continue
        idxs = clusters[cid]
        cluster_emb = embeddings[idxs].astype(np.float32)
        cluster_keys = [keys[i] for i in idxs]

        # Cuantizar
        q_emb = quantize_int8(cluster_emb)

        # Guardar
        np.save(f"{out_dir}/cluster_{cid:04d}.npy", q_emb)
        with open(f"{out_dir}/cluster_{cid:04d}.keys", "w") as f:
            json.dump(cluster_keys, f)
        print(f"  Cluster {cid:04d}: {len(idxs)} recs, {q_emb.nbytes / 1e6:.1f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build IVF index")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=4096)
    args = parser.parse_args()
    Path(args.out).mkdir(exist_ok=True)
    build_ivf(args.embeddings, args.keys, args.out, args.k)
