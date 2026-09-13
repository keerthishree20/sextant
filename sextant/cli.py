"""Command line interface.

    sextant demo                                   build a small index and show the curve
    sextant build --vectors v.npy --out index.npz
    sextant search --index index.npz --vector q.npy --k 10
    sextant recall --index index.npz --vectors v.npy --queries q.npy
    sextant stats  --index index.npz

Vectors are `.npy` files: a 2-D array for a corpus, 1-D or 2-D for queries. Ids
come from a text file, one per line, and default to the row number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from .brute import BruteForce
from .index import Index, recall_at_k


def _load_vectors(path: str) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise SystemExit(f"{path}: expected a 1-D or 2-D array, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _load_ids(path: str | None, count: int) -> list[str]:
    if path is None:
        return [str(i) for i in range(count)]
    ids = pathlib.Path(path).read_text().split()
    if len(ids) != count:
        raise SystemExit(f"{path}: {len(ids)} ids for {count} vectors")
    return ids


def cmd_build(args) -> int:
    vectors = _load_vectors(args.vectors)
    ids = _load_ids(args.ids, len(vectors))

    started = time.perf_counter()
    index = Index(vectors.shape[1], metric=args.metric, m=args.m,
                  ef_construction=args.ef_construction, ef_search=args.ef_search,
                  seed=args.seed)
    index.add_many(ids, vectors)
    elapsed = time.perf_counter() - started

    index.save(args.out)
    print(f"built {len(index):,} vectors of {index.dim} dimensions in {elapsed:.1f}s "
          f"({len(index) / elapsed:,.0f}/s)")
    print(f"wrote {args.out} ({pathlib.Path(args.out).stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_search(args) -> int:
    index = Index.load(args.index)
    for row, query in enumerate(_load_vectors(args.vector)):
        hits = index.search(query, k=args.k, ef=args.ef)
        if args.json:
            print(json.dumps([{"id": h.id, "distance": h.distance} for h in hits]))
            continue
        print(f"query {row}")
        for rank, hit in enumerate(hits, 1):
            print(f"  {rank:>3}. {hit.id:<24} {hit.distance:.6f}")
    return 0


def cmd_recall(args) -> int:
    """Measure this index against exact search on the same data."""
    index = Index.load(args.index)
    vectors = _load_vectors(args.vectors)
    queries = _load_vectors(args.queries)
    ids = _load_ids(args.ids, len(vectors))

    exact = BruteForce(index.dim, metric=index.metric)
    exact.add_many(ids, vectors)
    truth = [exact.search_ids(q, args.k) for q in queries]

    print(f"{'ef':>6} {'recall@' + str(args.k):>10} {'queries/s':>12}")
    for ef in [int(v) for v in args.ef_values.split(",")]:
        started = time.perf_counter()
        got = [[h.id for h in index.search(q, args.k, ef=ef)] for q in queries]
        elapsed = time.perf_counter() - started
        print(f"{ef:>6} {recall_at_k(got, truth, args.k):>10.3f} "
              f"{len(queries) / elapsed:>12,.0f}")
    return 0


def cmd_stats(args) -> int:
    stats = Index.load(args.index).stats()
    if args.json:
        print(json.dumps(stats))
        return 0
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"{key.ljust(width)}  {value}")
    return 0


def cmd_demo(args) -> int:
    """Everything the index does, on data generated here, in one command."""
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.n, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    ids = [f"v{i}" for i in range(args.n)]

    started = time.perf_counter()
    index = Index(args.dim, metric="cosine", m=16, ef_construction=200, seed=args.seed)
    index.add_many(ids, vectors)
    build_seconds = time.perf_counter() - started

    exact = BruteForce(args.dim, metric="cosine")
    exact.add_many(ids, vectors)
    truth = [exact.search_ids(q, 10) for q in queries]

    print(f"{args.n:,} vectors of {args.dim} dimensions, built in {build_seconds:.1f}s")
    print(f"{index.stats()['layers']} layers, {index.stats()['edges']:,} edges\n")
    print(f"{'ef':>6} {'recall@10':>10} {'queries/s':>12}")
    for ef in (10, 32, 64, 128, 256):
        started = time.perf_counter()
        got = [[h.id for h in index.search(q, 10, ef=ef)] for q in queries]
        elapsed = time.perf_counter() - started
        print(f"{ef:>6} {recall_at_k(got, truth, 10):>10.3f} {len(queries) / elapsed:>12,.0f}")

    started = time.perf_counter()
    for query in queries:
        exact.search(query, 10)
    brute = len(queries) / (time.perf_counter() - started)
    print(f"\nexact search for comparison: {brute:,.0f} queries/s at recall 1.000")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sextant", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build an index from a .npy file")
    p.add_argument("--vectors", required=True)
    p.add_argument("--ids", default=None, help="one id per line; defaults to row numbers")
    p.add_argument("--out", required=True)
    p.add_argument("--metric", choices=("cosine", "l2", "inner_product"), default="cosine")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=64)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("search", help="query an index")
    p.add_argument("--index", required=True)
    p.add_argument("--vector", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("recall", help="measure against exact search")
    p.add_argument("--index", required=True)
    p.add_argument("--vectors", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--ids", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef-values", default="10,32,64,128,256")
    p.set_defaults(fn=cmd_recall)

    p = sub.add_parser("stats", help="describe an index")
    p.add_argument("--index", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("demo", help="build a random index and show the recall curve")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
