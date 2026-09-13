"""Where the graph starts beating exact search, and where it does not.

This is the benchmark that decides whether the index is worth using at all. A
graph search costs roughly the same however large the corpus is; exact search
costs proportionally more. So there is a corpus size below which the index is
pure overhead, and finding it honestly matters more than any single throughput
figure.

For each size, the sweep finds the smallest `ef` reaching the recall target and
reports the speed there. Comparing an approximate index to exact search at any
other point is comparing two different answers.

    python -m bench.scaling --sizes 5000,20000,40000 --dim 128 --target 0.90
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sextant import BruteForce, Index, recall_at_k

EF_LADDER = [16, 32, 64, 128, 256, 512, 1024]


def make_data(n: int, dim: int, queries: int, seed: int, clustered: bool):
    """Two shapes of data, and they are not equally hard.

    Independent Gaussian coordinates in high dimensions are close to the worst
    case for any nearest-neighbour index: every pair of points sits at almost
    the same distance, so there is little structure for a graph to exploit.
    Real embeddings are nothing like that. They are clustered and their
    intrinsic dimension is far below their nominal one, which is exactly the
    structure the graph is built to use.
    """
    rng = np.random.default_rng(seed)
    if not clustered:
        return (rng.normal(size=(n, dim)).astype(np.float32),
                rng.normal(size=(queries, dim)).astype(np.float32))

    groups = max(n // 500, 8)
    centres = rng.normal(size=(groups, dim)).astype(np.float32) * 6
    assignment = rng.integers(0, groups, size=n)
    vectors = centres[assignment] + rng.normal(size=(n, dim)).astype(np.float32) * 0.6
    probe_at = rng.integers(0, groups, size=queries)
    probes = centres[probe_at] + rng.normal(size=(queries, dim)).astype(np.float32) * 0.6
    return vectors.astype(np.float32), probes.astype(np.float32)


def measure(n: int, dim: int, queries: int, k: int, target: float,
            m: int, ef_construction: int, seed: int, clustered: bool = False) -> dict:
    vectors, probes = make_data(n, dim, queries, seed, clustered)
    ids = [f"v{i}" for i in range(n)]

    started = time.perf_counter()
    index = Index(dim, metric="cosine", m=m, ef_construction=ef_construction, seed=seed)
    index.add_many(ids, vectors)
    build_seconds = time.perf_counter() - started

    exact = BruteForce(dim, metric="cosine")
    exact.add_many(ids, vectors)
    started = time.perf_counter()
    truth = [exact.search_ids(q, k) for q in probes]
    exact_qps = queries / (time.perf_counter() - started)

    reached = None
    for ef in EF_LADDER:
        started = time.perf_counter()
        got = [[hit.id for hit in index.search(q, k, ef=ef)] for q in probes]
        elapsed = time.perf_counter() - started
        recall = recall_at_k(got, truth, k)
        reached = {"ef": ef, "recall": round(recall, 4),
                   "queries_per_s": round(queries / elapsed, 1)}
        if recall >= target:
            break

    return {
        "vectors": n,
        "clustered": clustered,
        "build_seconds": round(build_seconds, 1),
        "built_per_s": round(n / build_seconds, 1),
        "exact_queries_per_s": round(exact_qps, 1),
        "at_target": reached,
        "target": target,
        "speedup": round(reached["queries_per_s"] / exact_qps, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.scaling", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default="5000,20000,40000")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--target", type=float, default=0.90)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--clustered", action="store_true",
                   help="clustered data, which is what real embeddings look like")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    rows = [measure(int(size), args.dim, args.queries, args.k, args.target,
                    args.m, args.ef_construction, args.seed, args.clustered)
            for size in args.sizes.split(",")]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    shape = "clustered" if args.clustered else "independent Gaussian"
    print(f"{args.dim} dimensions, {shape}, recall@{args.k} target {args.target}\n")
    print(f"{'vectors':>9} {'build':>9} {'ef':>5} {'recall':>8} "
          f"{'graph q/s':>11} {'exact q/s':>11} {'speedup':>9}")
    for row in rows:
        hit = row["at_target"]
        print(f"{row['vectors']:>9,} {row['build_seconds']:>8.0f}s {hit['ef']:>5} "
              f"{hit['recall']:>8.3f} {hit['queries_per_s']:>11,.0f} "
              f"{row['exact_queries_per_s']:>11,.0f} {row['speedup']:>8.2f}x")

    print("\nExact search slows down in proportion to the corpus; the graph barely")
    print("moves. Where the speedup column crosses 1.00 is where the index starts")
    print("earning its place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
