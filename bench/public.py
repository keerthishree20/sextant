"""Recall and speed on a public embedding set: GloVe word vectors.

The synthetic benchmarks showed that data shape decides everything: random
Gaussian vectors are near worst case, clustered ones are far easier. Real
embeddings are the case that matters, so this runs on GloVe, 25 dimensions,
cosine distance, as distributed by ann-benchmarks:

    curl -L -o bench/.data/glove-25-angular.hdf5 http://ann-benchmarks.com/glove-25-angular.hdf5
    python -m bench.public --n 20000

The full set is 1.18 million vectors, far beyond what this pure-Python index
builds in reasonable time, so a fixed random sample is indexed. The file's own
ground truth covers the full set and is wrong for a sample, so exact neighbours
are recomputed against the sample with brute force. The queries are the set's
own held-out test vectors, which are not in the sample.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from sextant import BruteForce, Index, recall_at_k

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_FILE = HERE / ".data" / "glove-25-angular.hdf5"
EF_VALUES = [10, 16, 32, 64, 128, 256]


def load(path: pathlib.Path, n: int, queries: int, seed: int):
    import h5py
    with h5py.File(path) as f:
        total = f["train"].shape[0]
        rows = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
        base = np.asarray(f["train"][rows], dtype=np.float32)
        probe = np.asarray(f["test"][:queries], dtype=np.float32)
        metric = f.attrs.get("distance", "angular")
    return base, probe, total, metric


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.public", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=pathlib.Path, default=DEFAULT_FILE)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if not args.file.exists():
        raise SystemExit(f"{args.file} not found. Download it with:\n  curl -L -o {args.file} "
                         "http://ann-benchmarks.com/glove-25-angular.hdf5")

    base, probe, total, metric = load(args.file, args.n, args.queries, args.seed)
    ids = [str(i) for i in range(len(base))]

    started = time.perf_counter()
    index = Index(base.shape[1], metric="cosine", m=16, ef_construction=200, seed=args.seed)
    index.add_many(ids, base)
    build_s = time.perf_counter() - started

    exact = BruteForce(base.shape[1], metric="cosine")
    exact.add_many(ids, base)
    started = time.perf_counter()
    truth = [exact.search_ids(q, args.k) for q in probe]
    exact_qps = len(probe) / (time.perf_counter() - started)

    curve = []
    for ef in EF_VALUES:
        started = time.perf_counter()
        got = [[h.id for h in index.search(q, args.k, ef=ef)] for q in probe]
        elapsed = time.perf_counter() - started
        curve.append({"ef": ef, "recall": round(recall_at_k(got, truth, args.k), 4),
                      "queries_per_s": round(len(probe) / elapsed, 1)})

    result = {
        "dataset": "glove-25-angular", "metric": metric, "full_size": total,
        "sampled": len(base), "queries": len(probe), "dim": base.shape[1],
        "build_seconds": round(build_s, 1), "exact_queries_per_s": round(exact_qps, 1),
        "curve": curve,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"GloVe, {base.shape[1]} dimensions, cosine: {len(base):,} vectors sampled from "
          f"{total:,}, {len(probe)} held-out queries")
    print(f"built in {build_s:.0f}s\n")
    print(f"{'ef':>6} {'recall@' + str(args.k):>11} {'queries/s':>11} {'vs exact':>9}")
    for row in curve:
        print(f"{row['ef']:>6} {row['recall']:>11.3f} {row['queries_per_s']:>11,.0f} "
              f"{row['queries_per_s'] / exact_qps:>8.2f}x")
    print(f"{'exact':>6} {1.0:>11.3f} {exact_qps:>11,.0f} {1.0:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
