# Sextant — Complete Project Guide

## Table of Contents
1. [What is Sextant?](#what-is-sextant)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [How HNSW Works](#how-hnsw-works)
5. [Architecture](#architecture)
6. [Code Walkthrough](#code-walkthrough)
7. [Using Sextant](#using-sextant)
8. [Testing Strategy](#testing-strategy)
9. [Benchmarks](#benchmarks)
10. [Extending Sextant](#extending-sextant)
11. [Troubleshooting](#troubleshooting)

---

## What is Sextant?

Sextant is a vector search index. You give it vectors, such as text embeddings, and it answers "which
stored vectors are nearest to this one?" quickly and approximately.

It implements HNSW, Hierarchical Navigable Small World graphs, from Malkov and Yashunin (2016). numpy
does the arithmetic. The graph, its layers, the neighbour selection, the search, deletion and saving
are all written here.

The project's most important finding is honest: **for small collections, plain exact search with one
numpy matrix multiply is faster than the graph.** The README measures where that stops being true.

---

## Quick Start

Requires Python 3.10 or newer and numpy. On this machine `python3` is 3.6, so the Makefile uses
`python3.12`.

```bash
make install        # .venv with numpy and pytest
make test           # the full suite
make demo           # builds a small random index and prints the recall curve
```

```python
import numpy as np
from sextant import Index

index = Index(dim=128, metric="cosine")
index.add("doc-1", np.random.rand(128).astype("float32"))
index.search(np.random.rand(128), k=10, ef=64)   # [Hit(id="doc-1", distance=...)]
```

---

## Core Concepts

### Nearest-neighbour search
Given a query vector, find the `k` stored vectors with the smallest distance to it.

### Exact versus approximate
Exact search compares the query with every stored vector. It is always right and costs time
proportional to the collection size. Approximate search skips most comparisons and may miss some
true neighbours.

### Recall@k
The fraction of the true `k` nearest neighbours that the index returned. `1.0` means perfect.

### Metrics
| metric | reported distance |
|---|---|
| `cosine` | cosine distance, 0 to 2. Vectors are normalised on insert |
| `l2` | euclidean distance |
| `inner_product` | negated inner product, so it can be negative |

Smaller is always nearer.

### The two knobs
- `ef_construction` controls how hard the build searches for good neighbours. Fixed once built.
- `ef` controls how hard a query searches. Change it per query. Higher means better recall and
  slower queries.

---

## How HNSW Works

**A proximity graph.** Every vector is a node linked to some of its nearest neighbours. To search,
start somewhere, repeatedly step to whichever neighbour is closer to the query, and stop when none
is. This is a greedy walk.

**Layers.** A flat graph takes many small steps to cross the space. HNSW stacks layers, each a
sparser sample of the one below. Search starts on the sparse top layer, takes a few long strides,
then drops down a layer at a time and finishes on layer zero, which contains every node.

**Who goes on which layer.** Each new node draws its top layer from an exponential distribution with
scale `1 / ln(m)`. That makes each layer roughly `1/m` the size of the one below, with no central
decision.

**Degree caps.** Each node keeps at most `m` links on upper layers and `2m` on layer zero, because
layer zero decides recall.

**Neighbour selection, the part that matters.** Linking a new node to its `m` nearest candidates
builds a bad graph. Inside a dense cluster all those links point the same way, and a search that
arrives there cannot leave. The paper's Algorithm 4 accepts a candidate only if it is closer to the
new node than to every neighbour already accepted. That spreads the links out. Rejected candidates
fill any slots left over.

**Pruning.** When an insert pushes an existing node over its degree cap, that node re-runs the
selection on its own links and may drop one. So about 16% of edges end up one-way, which the paper
does too. What must hold is that every node stays reachable, and the tests check that directly.

---

## Architecture

```
  Index  (sextant/index.py)
  string ids  <->  node numbers
  soft deletes, save/load, stats
        │
        ▼
  HNSW  (sextant/hnsw.py)
  _vectors   float32 array, capacity doubles when full
  _levels    top layer of each node
  _links     one dict per layer: node -> neighbour list
  _entry     entry point, always on the top layer
        │ distance calls
        ▼
  distance.py  cosine / euclidean / inner product, vectorised with numpy

  BruteForce  (sextant/brute.py)   exact search, the reference for every recall number
```

---

## Code Walkthrough

### `sextant/hnsw.py`
The graph itself.

| method | what it does |
|---|---|
| `HNSW(dim, metric=, m=16, ef_construction=200, seed=)` | validates settings. `m` must be at least 2 and `ef_construction` at least `m` |
| `add(vector)` | draws a level, descends from the top, searches each layer, links neighbours both ways, trims overfull nodes |
| `search(vector, k, ef=)` | descends greedily through upper layers, then a beam search of width `ef` on layer zero |
| `_draw_level()` | the exponential level draw |
| `_descend()` | greedy single-step walk on one upper layer |
| `_search_layer()` | beam search keeping the `ef` best candidates |
| `_select_neighbours()` | Algorithm 4, described above |
| `_trim()` | re-runs selection on a node pushed over its cap. Uses one precomputed distance matrix, which measured 24% faster |

### `sextant/index.py`
`Index` is the public API. It maps string ids to node numbers, handles replacing an existing id,
soft deletes, and saving to one `.npz` file. `recall_at_k()` compares results against exact search.

### `sextant/distance.py`
The three metrics, named `cosine`, `l2` and `inner_product`, as vectorised numpy functions, plus `normalise()` for cosine and `report()` to turn
internal ordering values into the distances users see.

### `sextant/brute.py`
`BruteForce` does exact search with one matrix multiply. Every recall figure is measured against it.

### `sextant/cli.py`
`demo`, `build`, `search`, `recall` and `stats` commands.

---

## Using Sextant

### API
| call | does |
|---|---|
| `Index(dim, metric="cosine", m=16, ef_construction=200, ef_search=64, seed=None)` | create |
| `add(id, vector)` | insert, or replace if the id exists |
| `add_many(ids, vectors)` | insert a 2-D array |
| `search(vector, k=10, ef=None)` | list of `Hit(id, distance)`, nearest first |
| `delete(id)` | hide an id. Returns `False` if absent |
| `get(id)`, `ids()`, `len(index)`, `id in index` | inspect |
| `save(path)`, `Index.load(path)` | one `.npz` file |
| `stats()` | nodes, deletions, layers, edges per layer |

NaN and infinite values are rejected when added, so they cannot poison later comparisons.

### Command line
```
sextant demo --n 5000 --dim 64
sextant build --vectors v.npy --out index.npz
sextant search --index index.npz --vector q.npy --k 10
sextant recall --index index.npz --vectors v.npy --queries q.npy
sextant stats --index index.npz
```

### Deletion is soft
A deleted id disappears from results, but its node stays in the graph as a waypoint. Removing it
would break the paths through it. Search widens itself in proportion to deletions so you still get a
full page of results. Rebuild the index when many ids are deleted.

### Choosing settings
1. Start with the defaults: `m=16`, `ef_construction=200`.
2. Measure recall with `sextant recall` against your own queries.
3. Raise `ef` until recall is high enough, and check whether exact search is faster at your size.

---

## Testing Strategy

| file | what it covers |
|---|---|
| `tests/test_distance.py` | the three metrics and normalisation |
| `tests/test_graph.py` | reachability, degree caps, no self-loops or duplicates, layer sizes, same seed same graph, NaN rejection |
| `tests/test_recall.py` | recall reaches 1.0 when `ef` covers the corpus, rises with `ef`, and survives the clustered-data trap |
| `tests/test_index.py` | ids, replacement, soft deletion, exact matches under every metric |
| `tests/test_persistence.py` | save and load give identical adjacency and identical results |

The clustered-data test builds six tight clusters far apart. A graph without Algorithm 4 traps the
search in whichever cluster it lands in.

---

## Benchmarks

```bash
make bench           # recall and speed against ef, one corpus size
make bench-scaling   # graph against exact search as the corpus grows. Slow
make bench-public    # real GloVe embeddings. Downloads 121 MB once and needs h5py
```

The full results are in the README. The summary to remember:

- On random Gaussian vectors exact search wins at every tested size.
- On clustered data and on real GloVe embeddings, the graph reaches high recall at a much lower
  `ef`, but exact search still wins at 20,000 vectors.
- A graph build in Python costs roughly 90 vectors per second at the default settings.

---

## Extending Sextant

Left out on purpose, as the README explains:

- **Product quantisation** to shrink memory.
- **A C extension** for the inner loops, which is where the measured time goes.
- **Filtered search** during traversal.
- **Incremental rebuild** to clear soft deletes.
- **A multi-threaded build.**

---

## Troubleshooting

### `ValueError: m must be at least 2`
Below 2 the graph cannot stay connected. Use the default of 16.

### Recall is low
Raise `ef` on the query first. If that is not enough, rebuild with a higher `ef_construction`. Check
the metric matches how your embeddings were trained. Most text embeddings expect cosine.

### Searches are slower than expected
For a few thousand vectors, exact search is genuinely faster. Compare with `BruteForce` before tuning.

### `make bench-public` fails on import
It needs `h5py`, which is not in the default install:
```bash
.venv/bin/pip install h5py
```

### Building takes minutes
Expected. Each insert runs a full layer search in Python. Build once and use `save()` and `load()`.
