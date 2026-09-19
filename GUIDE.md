# Sextant — Complete Project Guide

A complete guide from zero to a working vector search index. Covers every feature, every design
decision and the reason behind it, with the real code. It is self-contained: you can paste it into
any AI chat and ask questions about the project without sharing the repository.

**Repository:** https://github.com/keerthishree20/sextant

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Why](#2-tech-stack--why)
3. [Project Setup from Scratch](#3-project-setup-from-scratch)
4. [Core Ideas in Plain Words](#4-core-ideas-in-plain-words)
5. [Project Structure](#5-project-structure)
6. [Distance Metrics](#6-distance-metrics)
7. [The Layered Graph](#7-the-layered-graph)
8. [Choosing Each Node's Layer](#8-choosing-each-nodes-layer)
9. [Inserting a Vector](#9-inserting-a-vector)
10. [Neighbour Selection (the Part That Matters)](#10-neighbour-selection-the-part-that-matters)
11. [Pruning Overfull Nodes](#11-pruning-overfull-nodes)
12. [Searching](#12-searching)
13. [The Index: IDs, Deletion, Replacement](#13-the-index-ids-deletion-replacement)
14. [Saving and Loading](#14-saving-and-loading)
15. [Exact Search (the Reference)](#15-exact-search-the-reference)
16. [Measuring Recall](#16-measuring-recall)
17. [Python API](#17-python-api)
18. [Command Line](#18-command-line)
19. [Testing](#19-testing)
20. [Benchmarks & Results](#20-benchmarks--results)
21. [Two Optimisations, One of Which Failed](#21-two-optimisations-one-of-which-failed)
22. [Deliberately Not Built](#22-deliberately-not-built)
23. [Troubleshooting](#23-troubleshooting)
24. [Complete Feature Summary](#24-complete-feature-summary)

---

## 1. Project Overview

Sextant is a **vector search index**. You give it vectors, such as text embeddings from an AI model,
and it answers "which stored vectors are nearest to this one?" quickly and approximately. This is the
core of semantic search and of retrieval for AI chatbots (RAG).

It implements **HNSW** (Hierarchical Navigable Small World graphs) from Malkov and Yashunin, 2016.
numpy does the arithmetic. The graph, its layers, the neighbour selection, the search, deletion and
saving are all written here.

```python
import numpy as np
from sextant import Index

index = Index(dim=128, metric="cosine")
index.add("doc-1", embedding)
index.search(query, k=10, ef=64)     # [Hit(id="doc-1", distance=0.031), ...]
```

The project's most important finding is honest: **for small collections, plain exact search with one
numpy matrix multiply is faster than the graph.** Section 20 measures where that changes.

**Status:** complete. 65 tests pass. Also measured on real GloVe word embeddings.

---

## 2. Tech Stack & Why

| Technology | Role | Why We Chose It |
|---|---|---|
| **Python 3.10+** | Language | the graph algorithm is readable line by line |
| **numpy** | Arithmetic | distance calculations for many vectors in one call, in C |
| **float32 vectors** | Storage | half the memory of float64, plenty of precision for embeddings |
| **`.npz` files** | Saving | numpy's own format loads arrays without parsing |
| **heapq** | Search | two priority queues drive the best-first search |
| **pytest** | Tests | structural tests of the graph, not just recall numbers |
| **h5py** (optional) | Benchmark | reads the public GloVe dataset |

---

## 3. Project Setup from Scratch

```bash
git clone https://github.com/keerthishree20/sextant.git
cd sextant
make install        # .venv with numpy and pytest (uses python3.12; python3 here is 3.6)
make test           # all 65 tests
make demo           # builds a small random index and prints the recall curve
```

Benchmarks: `make bench`, `make bench-scaling` (slow), `make bench-public` (downloads 121 MB once,
needs `h5py`).

---

## 4. Core Ideas in Plain Words

| Idea | Meaning |
|---|---|
| **Vector** | a list of numbers, for example 128 of them, describing a piece of text or an image |
| **Nearest neighbours** | the stored vectors with the smallest distance to a query |
| **Exact search** | compare the query with every stored vector. Always right, slower as data grows |
| **Approximate search** | skip most comparisons. Much faster at scale, may miss a few true neighbours |
| **Recall@k** | the share of the true `k` nearest that the index returned. 1.0 is perfect |
| **Proximity graph** | each vector is a node linked to some of its nearest neighbours |
| **Greedy walk** | from any node, step to whichever neighbour is closer to the query; stop when none is |

### Why layers?
A greedy walk on a flat graph takes many small steps to cross the space. HNSW stacks layers, each a
sparser sample of the one below, like a road map with motorways on top and side streets at the
bottom. Search starts on the sparse top layer, takes a few long strides, then drops down a layer at
a time and finishes on layer zero, which holds every node.

---

## 5. Project Structure

```
sextant/
  distance.py  the three metrics, normalisation, and what distance a caller sees
  hnsw.py      HNSW: levels, insert, neighbour selection, pruning, layered search
  index.py     Index: string ids, deletion, save and load, recall helper
  brute.py     BruteForce: exact search, the reference for every recall figure
  cli.py       demo, build, search, recall, stats
bench/
  curve.py     recall and speed against ef, one corpus size
  scaling.py   graph against exact search as the corpus grows, two data shapes
  public.py    real GloVe word embeddings
tests/
  test_distance.py     metrics and normalisation
  test_graph.py        structural properties of the graph
  test_recall.py       recall curves and the clustered-data trap
  test_index.py        ids, replacement, deletion, every metric
  test_persistence.py  save and load round trip
Makefile  pyproject.toml  requirements.txt  requirements-dev.txt
```

---

## 6. Distance Metrics

Defined in `sextant/distance.py`. Three metrics, named `cosine`, `l2` and `inner_product`.

```python
def l2_squared(query, candidates):
    diff = candidates - query
    return np.einsum("...i,...i->...", diff, diff)   # no sqrt: it never changes the order

def negative_inner_product(query, candidates):
    return -(candidates @ query)

_FUNCTIONS = {
    "cosine": negative_inner_product,     # on unit vectors, cosine IS the inner product
    "inner_product": negative_inner_product,
    "l2": l2_squared,
}
```

### Why cosine reuses inner product
Cosine similarity is the inner product of unit-length vectors. So the index normalises vectors when
they are added and when a query arrives, and then runs exactly the same code path.

### What the caller sees: smaller is always nearer
| metric | reported distance |
|---|---|
| `cosine` | cosine distance, `1 - similarity`, from 0 to 2 |
| `l2` | euclidean distance (the square root is taken only for the result) |
| `inner_product` | negated inner product, so it can be negative. Inner product is not a true metric |

Zero vectors are left as zeros when normalising, instead of producing NaN.

---

## 7. The Layered Graph

`sextant/hnsw.py` stores:

| Field | Holds |
|---|---|
| `_vectors` | every vector in one float32 array. Capacity doubles when full |
| `_levels` | each node's top layer |
| `_links` | one dictionary per layer: node → list of neighbours on that layer |
| `_entry` | the entry point, always a node on the top layer |
| `m` | degree cap on upper layers (default 16) |
| `m0` | degree cap on layer zero, `2 * m` |

### Why layer zero gets twice the links
Every search finishes on layer zero, so its connectivity decides recall.

Settings are validated: `m` must be at least 2, and `ef_construction` at least `m`.

---

## 8. Choosing Each Node's Layer

```python
def _draw_level(self) -> int:
    return int(-math.log(max(self._rng.random(), 1e-12)) * self._level_scale)
# with self._level_scale = 1.0 / math.log(m)
```

An exponential draw with scale `1 / ln(m)`. Most nodes land on layer zero, and each layer is roughly
`1/m` the size of the one below. Nobody decides who goes on top; the randomness does it.

---

## 9. Inserting a Vector

```python
def add(self, vector) -> int:
    vector = self._prepare(vector)          # validate shape, reject NaN/inf, normalise for cosine
    node = self._count
    ...
    level = self._draw_level()
    if self._entry is None:                 # first node becomes the entry point
        self._entry, self._max_level = node, level
        return node

    cursor = self._entry
    for lc in range(self._max_level, level, -1):          # above our top layer: just walk downhill
        cursor = self._descend(vector, cursor, lc)

    for lc in range(min(level, self._max_level), -1, -1):  # on our layers: search and link
        found = self._search_layer(vector, [cursor], self.ef_construction, lc)
        capacity = self.m0 if lc == 0 else self.m
        chosen = self._select_neighbours(vector, found, capacity)
        self._links[lc][node] = list(chosen)
        for other in chosen:
            self._links[lc].setdefault(other, []).append(node)   # links are two-way
            self._trim(other, lc, capacity)
        if found:
            cursor = found[0][1]

    if level > self._max_level:             # a new tallest node becomes the entry point
        self._entry, self._max_level = node, level
    return node
```

### Why two-way links?
A one-way edge is invisible to any search arriving from the other side. That is how a graph ends up
with islands no search can reach.

---

## 10. Neighbour Selection (the Part That Matters)

The obvious approach, linking each new node to its `m` nearest candidates, **builds a bad graph**.
Inside a dense cluster all `m` links point the same way, the node gets no edge leading out of the
cluster, and a search that arrives there cannot leave.

The paper's Algorithm 4 fixes it: accept a candidate only if it is closer to the new node than to
anything already accepted. That forces the links to spread in different directions.

```python
def _select_neighbours(self, vector, candidates, capacity):
    if len(candidates) <= capacity:
        return [node for _dist, node in candidates]
    chosen, discarded = [], []
    for dist, node in sorted(candidates):
        if len(chosen) >= capacity:
            break
        if not chosen:
            chosen.append(node)
            continue
        to_chosen = self._distance(self._vectors[node], self._vectors[chosen])
        if dist < float(np.min(to_chosen)):
            chosen.append(node)            # closer to the new node than to anything chosen
        else:
            discarded.append((dist, node))
    for _dist, node in discarded:          # fill leftover slots with the nearest rejects
        if len(chosen) >= capacity:
            break
        chosen.append(node)
    return chosen
```

It is a dozen lines, and it is the difference between an index that works and one that looks fine on
uniform data and collapses on clustered data. A test builds six tight clusters far apart to prove it.

---

## 11. Pruning Overfull Nodes

When a new node links to an existing node, that node may go over its degree cap. `_trim` re-runs the
neighbour selection on its links and drops the extras.

### The one-way edge side effect
The node being trimmed may drop the edge it was just given, while the new node keeps its side. The
paper prunes the same way. Measured here, about **16% of edges end up one-way**. The test suite allows
this and bounds it, and separately checks that **every node stays reachable**, which is the property
that actually matters.

`_trim` uses one precomputed distance matrix for all its candidates (section 21 explains why here and
not in `_select_neighbours`).

---

## 12. Searching

```python
def search(self, vector, k, *, ef=None, skip=()):
    ef = max(ef if ef is not None else max(k, 32), k)     # exploring fewer than k cannot answer
    vector = self._prepare(vector)
    cursor = self._entry
    for lc in range(self._max_level, 0, -1):
        cursor = self._descend(vector, cursor, lc)         # greedy walk down the upper layers
    found = self._search_layer(vector, [cursor], ef, 0)    # wide search on layer zero
    ...
    return found[:k]
```

### The layer search: two heaps
```python
candidates = []   # min-heap: places still worth visiting, nearest first
results = []      # max-heap: the best ef found so far, worst on top
while candidates:
    nearest_dist, nearest = heapq.heappop(candidates)
    if -results[0][0] < nearest_dist and len(results) >= ef:
        break                     # nothing left can beat the worst result we hold
    unvisited = [n for n in links.get(nearest, ()) if n not in visited]
    ...
    for node, dist in zip(unvisited, self._distance(vector, self._vectors[unvisited])):
        ...                       # one numpy call for the whole neighbourhood
```

### Why one numpy call per neighbourhood?
This loop is where the time goes. Computing all of a node's neighbour distances in one call keeps
the per-edge work out of Python.

### The `ef` knob
`ef` is how many candidates the search keeps. Higher `ef` means better recall and slower queries, and
it can be changed on every query.

---

## 13. The Index: IDs, Deletion, Replacement

`sextant/index.py` wraps the graph with string ids.

- **`add(id, vector)`** inserts, or replaces the vector if the id already exists.
- **`delete(id)`** is **soft**: the id disappears from results but its node stays in the graph.

```python
def delete(self, id: str) -> bool:
    node = self._nodes.pop(id, None)
    if node is None:
        return False
    self._deleted.add(node)
    return True
```

### Why not remove the node?
Removing it would mean repairing every edge that pointed at it, and a graph that loses waypoints loses
the connectivity that makes search work. The cost: deleted vectors keep using memory and are still
walked through until you rebuild.

### Searching around deletions
```python
if self._deleted:
    live = max(len(self._nodes), 1)
    total = live + len(self._deleted)
    ef = min(int(ef * total / live) + k, total)     # widen in proportion to deletions
```
Without this, an index that is half deleted would quietly return half as many results as asked. A test
checks a full page still comes back.

NaN and infinity are rejected when added, so they can never poison later comparisons.

---

## 14. Saving and Loading

`save(path)` writes one `.npz` file:
- `vectors` and `levels` arrays,
- per layer, the adjacency lists as a flat array plus offsets (not JSON), which is much smaller and
  loads without parsing,
- metadata: format version, dim, metric, m, ef settings, entry point, max level, ids, deleted set.

`Index.load(path)` restores it. A test checks a round trip gives byte-identical adjacency and
identical search results.

---

## 15. Exact Search (the Reference)

`sextant/brute.py`:

```python
bf = BruteForce(dim=128, metric="cosine")
bf.add_many(ids, vectors)
bf.search(query, k=10)          # exact, one matrix multiply
```

Every recall number in the project is measured against it. It is also the honest competitor: at
small sizes it wins (section 20).

---

## 16. Measuring Recall

```python
from sextant.index import recall_at_k
recall_at_k(approximate_results, exact_results, k=10)
```

Or from the command line with `sextant recall`.

### How to tune
1. Start with the defaults: `m=16`, `ef_construction=200`.
2. Measure recall on your own queries.
3. Raise `ef` until recall is high enough.
4. Compare with `BruteForce` speed at your size. At a few thousand vectors, exact search may simply
   be faster.

---

## 17. Python API

| Call | Does |
|---|---|
| `Index(dim, metric="cosine", m=16, ef_construction=200, ef_search=64, seed=None)` | create |
| `add(id, vector)` | insert, or replace if the id exists |
| `add_many(ids, vectors)` | insert a 2-D array |
| `search(vector, k=10, ef=None)` | list of `Hit(id, distance)`, nearest first |
| `delete(id)` | hide an id; `False` if absent |
| `get(id)`, `ids()`, `len(index)`, `id in index` | inspect |
| `save(path)`, `Index.load(path)` | one `.npz` file |
| `stats()` | nodes, deletions, layers, edges per layer |

`ef_construction` is fixed at build time. `ef` can change on every query.

---

## 18. Command Line

```
sextant demo --n 5000 --dim 64           build a random index, print the recall curve
sextant build --vectors v.npy --out index.npz
sextant search --index index.npz --vector q.npy --k 10
sextant recall --index index.npz --vectors v.npy --queries q.npy
sextant stats --index index.npz
```

---

## 19. Testing

Recall reaching exactly 1.0 when `ef` covers the whole corpus is the headline check, but a graph can
score well and still be quietly broken. So the suite also checks the structure directly:

| Property | Why it matters |
|---|---|
| every node reachable from the entry point on layer zero | an unreachable node can never be returned |
| no node over its degree cap | `2m` on layer zero, `m` above |
| no self-loops, no duplicate edges | both waste degree |
| entry point on the top layer | otherwise the descent starts in the wrong place |
| layers thin out by about the branching factor | confirms the exponential level draw |
| edges two-way at least 84% of the time | pruning makes some one-way (section 11) |
| the same seed builds the same graph | makes every other test meaningful |

Also tested: exact matches under all three metrics, duplicate vectors (all distances zero), NaN and
infinity rejected, soft deletion returning a full page, the clustered-data trap, and save/load.

---

## 20. Benchmarks & Results

One machine: Intel Core i5-11320H, Python 3.12, numpy 2.5.

### Recall and speed against `ef`
20,000 random Gaussian vectors, 128 dimensions, `m=16`, `ef_construction=200`.

| `ef` | recall@10 | queries/s |
|---:|---:|---:|
| 10 | 0.163 | 1,533 |
| 32 | 0.379 | 737 |
| 64 | 0.561 | 328 |
| 128 | 0.762 | 158 |
| 256 | 0.929 | 83 |
| exact search | 1.000 | 2,423 |

**Exact search is faster than the graph at every recall level here.** That is the real result, not a
bug. Exact search is one matrix multiply that runs near the machine's memory speed. The graph makes
hundreds of small numpy calls per query, each with Python overhead.

### Where the graph catches up
| Data | Vectors | `ef` for recall 0.90 | recall | Graph q/s | Exact q/s | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 5,000 | 128 | 0.969 | 527 | 12,373 | 0.04× |
| Gaussian | 20,000 | 256 | 0.932 | 162 | 2,196 | 0.07× |
| Gaussian | 40,000 | 512 | 0.950 | 83 | 739 | 0.11× |
| Clustered | 20,000 | **16** | 0.949 | 1,499 | 2,041 | **0.73×** |

Random Gaussian vectors are close to the worst case for any index: points sit at almost identical
distances from each other. Real embeddings are clustered, and there the graph needs `ef` 16 instead
of 256, and the crossover moves from about a million vectors to tens of thousands.

### Real embeddings: GloVe
20,000 vectors sampled from GloVe (25 dimensions, cosine), queries from the dataset's own test set.

| `ef` | recall@10 | queries/s | vs exact |
|---:|---:|---:|---:|
| 10 | 0.895 | 1,632 | 0.61× |
| 16 | 0.937 | 1,399 | 0.53× |
| 64 | 0.996 | 529 | 0.20× |
| exact | 1.000 | 2,659 | 1.00× |

Real data behaves like the clustered set. Exact search still wins at this size.

### Build cost
About 90 vectors a second at the default settings, so 40,000 vectors took 505 seconds.

---

## 21. Two Optimisations, One of Which Failed

**Failed:** precomputing the whole candidate-to-candidate distance matrix in `_select_neighbours` made
building 4,000 vectors **56% slower** (57 s to 100 s). The loop usually stops after `capacity`
acceptances, so most of the matrix is computed and never read.

**Worked:** the same trick inside `_trim`, where the candidate list is barely larger than the cap and
nearly all the matrix is read. Build time dropped from 57 s to **43 s**, 24% faster.

Same idea, opposite results, decided by measurement both times.

---

## 22. Deliberately Not Built

| Feature | Why not |
|---|---|
| product quantisation | compressing vectors is the next memory win, and a separate project |
| a C extension | would move the crossover more than any algorithm change left |
| filtered search | restricting by metadata during the walk changes the algorithm |
| incremental rebuild | deletions build up until you rebuild by hand |
| multi-threaded build | inserts take a graph-wide lock by design here |

---

## 23. Troubleshooting

### `ValueError: m must be at least 2`
Below 2 the graph cannot stay connected. Use the default 16.

### Recall is low
Raise `ef` first. If that is not enough, rebuild with a higher `ef_construction`. Check the metric
matches how your embeddings were trained; most text embeddings expect `cosine`.

### Search is slower than expected
For a few thousand vectors, exact search really is faster. Compare with `BruteForce`.

### `make bench-public` fails on import
```bash
.venv/bin/pip install h5py
```

### Building takes minutes
Expected. Build once, then `save()` and `load()`.

---

## 24. Complete Feature Summary

### All Features Built

| # | Feature | Type | Key Files |
|---|---|---|---|
| 1 | Cosine, L2 and inner-product metrics | Math | `distance.py` |
| 2 | Exponential layer assignment | Graph | `hnsw.py` |
| 3 | Layered insert with two-way links | Graph | `hnsw.py` |
| 4 | Heuristic neighbour selection (Algorithm 4) | Graph | `hnsw.py` |
| 5 | Pruning with a precomputed matrix | Graph | `hnsw.py` |
| 6 | Greedy descent plus beam search | Search | `hnsw.py` |
| 7 | String ids and replacement | Index | `index.py` |
| 8 | Soft deletion with widened search | Index | `index.py` |
| 9 | Compact `.npz` save and load | Index | `index.py` |
| 10 | Exact search reference | Baseline | `brute.py` |
| 11 | Recall measurement | Tooling | `index.py`, `cli.py` |
| 12 | Command line | Tooling | `cli.py` |
| 13 | Structural graph tests | Testing | `tests/test_graph.py` |
| 14 | Synthetic, scaling and GloVe benchmarks | Tooling | `bench/` |

### Data Flow Architecture

```
add(id, vector)
  └── validate + normalise ──► draw level ──► walk down from the entry point
        └── on each of the node's layers:
              search_layer(ef_construction) ──► select_neighbours ──► link both ways ──► trim overfull

search(vector, k, ef)
  └── normalise ──► greedy descent on upper layers ──► search_layer(ef) on layer 0
        └── widen ef for deletions ──► skip deleted ──► top k ──► Hit(id, distance)

save(path) ──► vectors + levels + flat adjacency per layer + metadata ──► one .npz
```

### Tech Stack at a Glance

```
Language:  Python 3.10+
Math:      numpy (float32, einsum, matrix multiply)
Algorithm: HNSW (Malkov & Yashunin 2016) with Algorithm 4 neighbour selection
Storage:   numpy .npz
Testing:   pytest, structural graph checks, recall curves
Data:      synthetic Gaussian, synthetic clustered, GloVe (ann-benchmarks)
```
