"""
SHARD Command-Line Interface

Usage:
    python -m shard build  --input data.json --output ./mydb [--shards 1000]
    python -m shard query  --db ./mydb --key "ababol"
    python -m shard search --db ./mydb --query "planta del campo" [--top-k 5]
    python -m shard stats  --db ./mydb
"""

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_build(args) -> None:
    from shard.storage.shard_writer import ShardWriter
    from shard.index.index_builder import IndexBuilder

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: input JSON must be an array of objects.", file=sys.stderr)
        sys.exit(1)

    print(f"Building SHARD database...")
    print(f"  Input   : {input_path} ({len(data):,} records)")
    print(f"  Output  : {args.output}")
    print(f"  Shards  : {args.shards}")
    print(f"  Key     : {args.key_field}")
    print(f"  Value   : {args.value_field}")

    start = time.time()

    with ShardWriter(args.output, num_shards=args.shards) as writer:
        builder = IndexBuilder(
            args.output,
            num_shards=args.shards,
            num_hashes=args.num_hashes,
        )

        for i, record in enumerate(data):
            key = str(record.get(args.key_field, f"record_{i}"))
            value = json.dumps(record, ensure_ascii=False)
            writer.write(key, value)

            text = f"{key} {record.get(args.value_field, '')}"
            builder.add(i, key, text)

        builder.build()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.2f}s. Wrote {writer.total_written:,} records.")


def cmd_query(args) -> None:
    from shard.storage.mmap_reader import MMapReader

    with MMapReader(args.db, num_shards=args.shards) as reader:
        result = reader.find(args.key)

    if result is None:
        print(f"Key not found: {args.key!r}", file=sys.stderr)
        sys.exit(1)

    print(result)


def cmd_search(args) -> None:
    from shard.index.index_reader import IndexReader

    reader = IndexReader(args.db)
    reader.load()
    results = reader.lookup(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    for key, score in results:
        print(f"{score:.4f}  {key}")


def cmd_build_ivf(args) -> None:
    import numpy as np
    from shard.index.ivfpq_builder import build_ivfpq

    vectors = np.load(args.embeddings, mmap_mode="r")          # never fully loaded
    with open(args.keys, encoding="utf-8") as f:
        raw = json.load(f)
    keys = raw if isinstance(raw, list) else [raw[str(i)] for i in range(len(raw))]
    if len(keys) != len(vectors):
        print(f"Error: keys ({len(keys)}) != embeddings ({len(vectors)})", file=sys.stderr)
        sys.exit(1)
    build_ivfpq(vectors, keys, args.out, profile=args.profile, rerank=args.rerank, seed=args.seed)


def cmd_search_ivf(args) -> None:
    import numpy as np
    from shard.index.ivfpq_reader import IVFPQReader

    q = np.load(args.query_vec).astype(np.float32).reshape(-1)  # precomputed query vector
    reader = IVFPQReader(args.ivf)
    results = reader.search(q, top_k=args.top_k, nprobe=args.nprobe)
    if not results:
        print("No results found.")
        return
    for key, score in results:
        print(f"{score:.4f}  {key}")


def cmd_stats(args) -> None:
    db_path = Path(args.db)
    meta_path = db_path / "index.meta.json"

    shard_files = list(db_path.glob("shard_*.bin"))
    bloom_files = list(db_path.glob("shard_*.bloom"))
    total_shard_bytes = sum(f.stat().st_size for f in shard_files)
    total_bloom_bytes = sum(f.stat().st_size for f in bloom_files)

    print(f"SHARD database: {db_path.resolve()}")
    print(f"  Shard files   : {len(shard_files):,}")
    print(f"  Bloom filters : {len(bloom_files):,}")
    print(f"  Shard data    : {total_shard_bytes / 1e6:.2f} MB")
    print(f"  Bloom data    : {total_bloom_bytes / 1e6:.2f} MB")
    print(f"  Total on disk : {(total_shard_bytes + total_bloom_bytes) / 1e6:.2f} MB")

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  Total records : {meta['total_records']:,}")
        print(f"  Shards config : {meta['num_shards']:,}")
        print(f"  MinHash size  : {meta['num_hashes']} hashes")
    else:
        print("  (No index metadata found — build the index to enable similarity search)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shard",
        description="SHARD — Scalable Hash-Addressed Retrieval Database CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── build ──────────────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Build a SHARD database from a JSON file")
    p_build.add_argument("--input", required=True, metavar="FILE", help="Input JSON file (array of objects)")
    p_build.add_argument("--output", required=True, metavar="DIR", help="Output directory for the database")
    p_build.add_argument("--shards", type=int, default=1000, metavar="N", help="Number of shards (default: 1000)")
    p_build.add_argument("--key-field", default="lemma", dest="key_field", metavar="FIELD", help="JSON field to use as key (default: lemma)")
    p_build.add_argument("--value-field", default="definition", dest="value_field", metavar="FIELD", help="JSON field for index text (default: definition)")
    p_build.add_argument("--num-hashes", type=int, default=128, dest="num_hashes", metavar="H", help="MinHash signature size (default: 128)")

    # ── query ──────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Exact key lookup in a SHARD database")
    p_query.add_argument("--db", required=True, metavar="DIR", help="SHARD database directory")
    p_query.add_argument("--key", required=True, metavar="KEY", help="Key to look up")
    p_query.add_argument("--shards", type=int, default=1000, metavar="N")

    # ── search ─────────────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Semantic similarity search")
    p_search.add_argument("--db", required=True, metavar="DIR", help="SHARD database directory")
    p_search.add_argument("--query", required=True, metavar="TEXT", help="Query text")
    p_search.add_argument("--top-k", type=int, default=5, dest="top_k", metavar="K", help="Number of results (default: 5)")

    # ── build-ivf ────────────────────────────────────────────────────────────--
    p_bivf = sub.add_parser("build-ivf", help="Build an IVF-PQ vector index (offline)")
    p_bivf.add_argument("--embeddings", required=True, metavar="NPY", help="embeddings.npy (N,dim) L2-normalized")
    p_bivf.add_argument("--keys", required=True, metavar="JSON", help="JSON list of N keys (or {idx:key} map)")
    p_bivf.add_argument("--out", required=True, metavar="DIR", help="output ivf/ directory")
    p_bivf.add_argument("--profile", default="low-ram", choices=["low-ram", "medium", "fast"])
    p_bivf.add_argument("--rerank", default="auto", choices=["auto", "none", "sq8", "f32"])
    p_bivf.add_argument("--seed", type=int, default=0)

    # ── search-ivf ───────────────────────────────────────────────────────────--
    p_sivf = sub.add_parser("search-ivf", help="Vector search over an IVF-PQ index")
    p_sivf.add_argument("--ivf", required=True, metavar="DIR", help="ivf/ index directory")
    p_sivf.add_argument("--query-vec", required=True, dest="query_vec", metavar="NPY",
                        help="precomputed query vector (.npy, shape (dim,))")
    p_sivf.add_argument("--top-k", type=int, default=5, dest="top_k", metavar="K")
    p_sivf.add_argument("--nprobe", type=int, default=None, metavar="N", help="lists to probe (default: profile)")

    # ── stats ──────────────────────────────────────────────────────────────────
    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.add_argument("--db", required=True, metavar="DIR", help="SHARD database directory")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "build-ivf":
        cmd_build_ivf(args)
    elif args.command == "search-ivf":
        cmd_search_ivf(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
