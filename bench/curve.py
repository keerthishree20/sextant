"""The recall and speed curve, plus the exact search it has to beat.

An approximate index has exactly one number worth quoting, and it is a pair:
queries per second **at** a stated recall. Either half on its own is
meaningless. Returning wrong answers is always fast, and reading every vector is
always perfectly accurate.

    python -m bench.curve --n 20000 --dim 128
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sextant import BruteForce, Index, recall_at_k


def build(vectors: np.ndarray, ids, m: int, ef_construction: int, metric: str, seed: int):
    started = time.perf_counter()
    index = Index(vectors.shape[1], metric=metric, m=m,
                  ef_construction=ef_construction, seed=seed)
    index.add_many(ids, vectors)
    return index, time.perf_counter() - started


def sweep(index: Index, queries: np.ndarray, truth, k: int, ef_values) -> list[dict]:
    rows = []
    for ef in ef_values:
        started = time.perf_counter()
        got = [[hit.id for hit in index.search(q, k, ef=ef)] for q in queries]
        elapsed = time.perf_counter() - started
        rows.append({
            "ef": ef,
            "recall": round(recall_at_k(got, truth, k), 4),
            "queries_per_s": round(len(queries) / elapsed, 1),
            "milliseconds_per_query": round(elapsed / len(queries) * 1000, 3),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.curve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-values", default="10,16,32,64,128,256")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.n, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    ids = [f"v{i}" for i in range(args.n)]

    index, build_seconds = build(vectors, ids, args.m, args.ef_construction,
                                 args.metric, args.seed)

    exact = BruteForce(args.dim, metric=args.metric)
    exact.add_many(ids, vectors)
    started = time.perf_counter()
    truth = [exact.search_ids(q, args.k) for q in queries]
    brute_qps = len(queries) / (time.perf_counter() - started)

    rows = sweep(index, queries, truth, args.k, [int(v) for v in args.ef_values.split(",")])
    stats = index.stats()
    result = {
        "vectors": args.n,
        "dim": args.dim,
        "metric": args.metric,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "build_seconds": round(build_seconds, 2),
        "vectors_per_second_built": round(args.n / build_seconds, 1),
        "layers": stats["layers"],
        "edges": stats["edges"],
        "edges_per_node": round(stats["edges"] / args.n, 1),
        "exact_queries_per_s": round(brute_qps, 1),
        "curve": rows,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"{args.n:,} vectors, {args.dim} dimensions, metric {args.metric}")
    print(f"built in {result['build_seconds']}s "
          f"({result['vectors_per_second_built']:,.0f}/s), "
          f"{result['layers']} layers, {result['edges_per_node']} edges per node\n")
    print(f"{'ef':>6} {'recall@' + str(args.k):>11} {'queries/s':>12} {'ms/query':>10} {'speedup':>9}")
    for row in rows:
        print(f"{row['ef']:>6} {row['recall']:>11.3f} {row['queries_per_s']:>12,.0f} "
              f"{row['milliseconds_per_query']:>10.3f} "
              f"{row['queries_per_s'] / brute_qps:>8.1f}x")
    print(f"\nexact  {1.0:>11.3f} {brute_qps:>12,.0f} "
          f"{1000 / brute_qps:>10.3f} {1.0:>8.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
