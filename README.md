# SHARD — Scalable Hash-Addressed Retrieval Database

> **TB-scale knowledge on 2 GB RAM.** Store petabytes of structured data and query it in milliseconds on a Raspberry Pi.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

SHARD is a purpose-built binary database engine for massive, read-heavy datasets on resource-constrained hardware. It replaces JSON files, SQLite, and vector databases for use cases where you need to store millions or billions of records and query them instantly — without loading the full dataset into RAM.

Designed as the native storage backend for **[DASA](https://github.com/angelgabrieljacintohuayllasco/DASA)**.

Beyond exact-key lookup, SHARD now ships a **numpy-only IVF-PQ vector index** for
approximate nearest-neighbor search — fast semantic search over hundreds of
millions of embeddings while keeping query RAM in the hundreds of MB. No FAISS,
no server, mmap-backed, scales toward 1 TB on low-RAM hardware.

---

## The Problem with Standard Storage

| Format | 1 TB dataset on 2 GB RAM | Query speed |
|---|---|---|
| JSON array | **Impossible** — requires full load | O(N) scan after load |
| SQLite | Needs 500 MB+ overhead | OK for exact match |
| Vector DB (Qdrant, Weaviate) | Requires separate server | Fast, but heavy |
| PostgreSQL | 200+ MB RAM overhead | Fast, needs server |
| **SHARD** | **✓ Works natively** | O(1) key lookup + IVF-PQ vector search |

---

## How It Works

```
Key: "ababol"
    │
    ▼  FNV1a hash (deterministic, cross-platform)
hash = 12938472634
    │
    ▼  hash % num_shards
shard_id = 384
    │
    ├──▶ BloomFilter[384].contains("ababol")?
    │         NO → return None (zero disk I/O)
    │         YES → open shard_000384.bin
    │
    ▼  Memory-mapped linear scan (OS pages in only needed blocks)
Record found: {"lemma": "ababol", "definition": "Planta de la familia..."}
```

**Key insight:** The entire 1 TB dataset never touches RAM. The OS loads only the 4–64 KB page that contains your record via `mmap`. The rest stays on disk.

---

## Quick Start

### Build a database from JSON

```bash
python -m shard build --input my_dictionary.json --output ./mydb --shards 1000
```

### Query a record

```bash
python -m shard query --db ./mydb --key "ababol" --shards 1000
```

### Similarity search

```bash
python -m shard search --db ./mydb --query "planta del campo" --top-k 5
```

### Database statistics

```bash
python -m shard stats --db ./mydb
```

### In Python

```python
from shard.storage.mmap_reader import MMapReader

with MMapReader("./mydb", num_shards=1000) as reader:
    result = reader.find("ababol")
    print(result)  # {"lemma": "ababol", "definition": "Planta de la familia..."}
```

### MinHash similarity search

```python
from shard.index.index_reader import IndexReader

reader = IndexReader("./mydb")
reader.load()

results = reader.lookup("planta del campo", top_k=5)
for key, score in results:
    print(f"{score:.3f}  {key}")
```

---

## Vector search (IVF-PQ)

For semantic search over embeddings, build an **IVF-PQ index**: IVF (coarse
k-means) jumps straight to the few relevant clusters — like a database index
jumps to a row — and PQ (product quantization on residuals) compresses each
vector ~32× so the index is mmap-backed and tiny in RAM.

**Build is offline** (run it on a powerful machine / Colab), then copy the
read-only `ivf/` artifact to a low-RAM device for query. The corpus is never
re-embedded on the query device.

```bash
# embeddings.npy: (N, dim) float32, L2-normalized;  keys.json: list of N keys
python -m shard.cli build-ivf --embeddings emb.npy --keys keys.json \
    --out ./mydb/ivf --profile low-ram        # low-ram | medium | fast

# query with a precomputed query vector (.npy, shape (dim,))
python -m shard.cli search-ivf --ivf ./mydb/ivf --query-vec q.npy --top-k 5
```

```python
from shard.index.ivfpq_builder import build_ivfpq
from shard.index.ivfpq_reader import IVFPQReader
import numpy as np

# build (offline): vectors may be an np.memmap — never fully loaded into RAM
build_ivfpq(vectors, keys, "./mydb/ivf", profile="low-ram")

# query (low-RAM device): only centroids + codebooks + probed lists touch RAM
reader = IVFPQReader("./mydb/ivf")
for key, score in reader.search(query_vec, top_k=5):
    print(f"{score:.4f}  {key}")
```

**Profiles** trade recall, index size and speed. PQ alone ranks near-ties
poorly, so each profile keeps a disk-backed rerank cache (`sq8` int8 or `f32`)
that re-scores only the shortlist — high recall without breaking the low-RAM
budget.

| Profile | PQ bytes/vec | rerank cache | use when |
|---|---|---|---|
| `low-ram` | 48 | sq8 (384 B/vec) | smallest RAM, runs on 2 GB |
| `medium`  | 32 | sq8 (384 B/vec) | balanced |
| `fast`    | 16 | f32 (1536 B/vec) | highest recall, small/mid N |

---

## Input Format

SHARD accepts any JSON array:

```json
[
  {"lemma": "ababol", "definition": "Planta de la familia de las compuestas."},
  {"lemma": "ábaco",  "definition": "Instrumento de cálculo con barras y cuentas."}
]
```

Keys and values can be any UTF-8 string. The `--key-field` and `--value-field` CLI flags select which JSON fields to use.

---

## Project Structure

```
shard/
├── core/
│   ├── hasher.py          # MinHash + SimHash — compact similarity fingerprints
│   ├── sharding.py        # ShardRouter — maps keys to shard files via FNV1a
│   └── bloom_filter.py    # BloomFilter — eliminates unnecessary disk reads
├── storage/
│   ├── binary_encoder.py  # SHARD binary format: encode/decode records + CRC32
│   ├── shard_writer.py    # Streaming writer — builds .bin shards + .bloom files
│   └── mmap_reader.py     # Memory-mapped reader — O(1) shard + linear scan
├── index/
│   ├── ivfpq_format.py    # IVF-PQ profiles, params, portable manifest I/O
│   ├── ivfpq_builder.py   # IVF-PQ streaming build (numpy + MiniBatchKMeans)
│   ├── ivfpq_reader.py    # IVF-PQ query — mmap, ADC, sq8/f32 rerank
│   ├── index_builder.py   # Builds MinHash similarity index
│   ├── index_reader.py    # Queries the index for nearest-neighbor search
│   ├── ivf_builder.py     # DEPRECATED — superseded by ivfpq_builder
│   ├── ivf_reader.py      # DEPRECATED — superseded by ivfpq_reader
│   ├── tfidf_writer.py    # Builds TF-IDF posting lists (keyword search)
│   └── tfidf_reader.py    # BM25 keyword search over TF-IDF index
└── cli.py                 # Command-line interface

docs/
├── format-spec.md         # Binary record format specification
├── sharding.md            # Sharding algorithm and tuning guide
└── indexing.md            # MinHash index: build, query, and RAM usage

examples/
├── build_from_json.py     # Build a demo SHARD database from JSON
└── query_example.py       # Exact lookup + semantic similarity search
```

---

## Binary Format

Each record in a `.bin` shard file:

```
┌─────────────────────────────────────────────────────────────┐
│  4 bytes  │  record_length (big-endian uint32)               │
├───────────┼─────────────────────────────────────────────────┤
│  2 bytes  │  key_length (big-endian uint16)                  │
├───────────┼─────────────────────────────────────────────────┤
│  N bytes  │  key (UTF-8)                                     │
├───────────┼─────────────────────────────────────────────────┤
│  4 bytes  │  value_length (big-endian uint32)                │
├───────────┼─────────────────────────────────────────────────┤
│  M bytes  │  value (UTF-8 — typically JSON)                  │
├───────────┼─────────────────────────────────────────────────┤
│  4 bytes  │  CRC32 checksum                                  │
└─────────────────────────────────────────────────────────────┘
```

See [docs/format-spec.md](docs/format-spec.md) for full specification.

---

## Capacity Guidelines

| Dataset size | Recommended shards | RAM for Bloom filters | RAM for MinHash index | RAM for IVF-PQ query |
|---|---|---|---|---|
| 10k records | 16 | ~1 MB | ~6 MB | ~5 MB |
| 1M records | 1 000 | ~12 MB | ~512 MB | ~10 MB |
| 100M records | 10 000 | ~1.2 GB | ~50 GB* | ~110 MB |
| 1B records | 100 000 | ~12 GB* | Not recommended | ~410 MB |

*At very large scales, use multiple SHARD nodes without building the full in-RAM index.

**IVF-PQ query RAM** counts only the resident centroids + codebooks + currently
probed lists — the PQ codes and rerank cache stay on disk (mmap), so even a 1 B
index queries within a few hundred MB. This is the path for semantic search at
scale; the MinHash in-RAM index does not scale past ~10M.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

[Apache 2.0](LICENSE) — Free to use, modify and distribute.

---

*SHARD was born from a simple question: why does a Raspberry Pi need to load a 1 TB dictionary into RAM just to look up one word? It shouldn't.*
