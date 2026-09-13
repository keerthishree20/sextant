# Sextant

An HNSW vector index with the graph written from scratch. numpy does the
arithmetic; the layers, the neighbour selection, the search and the index are
all here.

```python
import numpy as np
from sextant import Index

index = Index(dim=128, metric="cosine")
index.add("doc-1", embedding)
index.search(query, k=10, ef=64)     # [Hit(id="doc-1", distance=0.031), ...]
```

---

## The idea

Build a proximity graph where every node links to its nearest neighbours, and
search becomes greedy walking: step to whichever neighbour is closer to the
query, stop when none is. That works, but a greedy walk on a flat graph takes a
long time to cross the space. So stack the graph into layers, each a sparser
sample of the one below, and start at the top. The upper layers cross the space
in a few long strides; the bottom layer does the fine approach.

Each node's layer count is an exponential draw with scale `1 / ln(m)`, which is
what makes the upper layers sparse without anyone deciding who goes in them.

From Malkov and Yashunin, *Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs* (2016).

### The part that matters

Neighbour selection. The obvious approach is to link each new node to the `m`
nearest candidates, and it builds a bad graph. Inside a dense cluster all `m` of
those links point the same way, the node ends up with no edge leading out of the
cluster, and a greedy search that arrives there cannot leave.

The paper's Algorithm 4 accepts a candidate only if it is closer to the new node
than to anything already accepted. That forces the links to spread out. It is a
dozen lines, it is the difference between a working index and one that scores
acceptably on uniform data and collapses on clustered data, and it is the first
thing a naive implementation leaves out.

The test suite has a case for exactly this: six tight clusters far apart, where
a graph without the heuristic traps the search in whichever cluster it lands in.

---

## Measured

Everything below is one machine: Intel Core i5-11320H, 8 threads, 15 GB RAM,
Linux 6.8, Python 3.12.13, numpy 2.5.3. Reproduce with `make bench`,
`make bench-scaling` and `make bench-public`.

### The recall and speed curve

20,000 vectors, 128 dimensions, independent Gaussian coordinates, `m=16`,
`ef_construction=200`.

| `ef` | recall@10 | queries/s |
|---:|---:|---:|
| 10 | 0.163 | 1,533 |
| 16 | 0.234 | 1,392 |
| 32 | 0.379 | 737 |
| 64 | 0.561 | 328 |
| 128 | 0.762 | 158 |
| 256 | 0.929 | 83 |
| exact search | 1.000 | 2,423 |

Read the last row before anything else. **Exact search is faster than the index
at every recall level here.** That is the real result, and it is not a bug in
the graph.

### Why, and where it stops being true

Exact search in numpy is a single matrix multiply, which BLAS runs at close to
the memory bandwidth of the machine. The graph makes hundreds of small numpy
calls per query, each with a microsecond or so of Python overhead that has
nothing to do with the arithmetic. At twenty thousand vectors the matmul is
still cheaper than the overhead.

Exact search costs proportionally more as the corpus grows. The graph costs far
less. So there is a crossover, and finding it honestly is the point:

| Vectors | `ef` for recall 0.90 | recall | Graph q/s | Exact q/s | Ratio |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 128 | 0.969 | 527 | 12,373 | 0.04x |
| 20,000 | 256 | 0.932 | 162 | 2,196 | 0.07x |
| 40,000 | 512 | 0.950 | 83 | 739 | 0.11x |

The ratio is climbing but slowly. Extrapolating that trend, this implementation
would need somewhere around a million vectors before it beat numpy on this data.

### Except that the data was the problem

Independent Gaussian coordinates in 128 dimensions are close to the worst case
for any nearest-neighbour index. In high dimensions such points sit at almost
identical distances from one another, so there is barely any structure for a
graph to exploit, and `ef` has to keep rising to hold recall.

Real embeddings are nothing like that. They are clustered, and their intrinsic
dimension is far below their nominal one. Same code, same size, same recall
target, clustered data:

| Data | Vectors | `ef` for recall 0.90 | recall | Graph q/s | Exact q/s | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 20,000 | 256 | 0.932 | 162 | 2,196 | 0.07x |
| Clustered | 20,000 | **16** | 0.949 | 1,499 | 2,041 | **0.73x** |

Sixteen instead of two hundred and fifty six. The ratio improves by a factor of
ten, and the crossover moves from about a million vectors to somewhere in the
tens of thousands.

The honest summary: this index is a real HNSW implementation whose recall curve
is correct, and in pure Python it is worth using at a few tens of thousands of
realistically shaped vectors and up. Below that, and on adversarial data, a
matrix multiply wins. Any vector index that quotes throughput without saying
which of those situations it was measured in is not telling you much.

### And on real embeddings

The synthetic results only matter if real data behaves like one of them. So the
same index was run on GloVe word vectors, 25 dimensions, cosine distance, from
the public ann-benchmarks collection. A fixed random sample of 20,000 vectors was
indexed from the 1,183,514 in the set. The set's own held-out test vectors were
the queries. Exact neighbours were recomputed against the sample, because the
file's ground truth covers the full set.

| `ef` | recall@10 | queries/s | vs exact |
|---:|---:|---:|---:|
| 10 | 0.895 | 1,632 | 0.61x |
| 16 | 0.937 | 1,399 | 0.53x |
| 32 | 0.980 | 800 | 0.30x |
| 64 | 0.996 | 529 | 0.20x |
| 128 | 0.999 | 295 | 0.11x |
| 256 | 1.000 | 161 | 0.06x |
| exact search | 1.000 | 2,659 | 1.00x |

Real embeddings behave like the clustered data, not the random data. Recall
passes 0.93 at `ef` 16, the same point the clustered synthetic set reached, where
random Gaussian vectors needed 256. Exact search still wins at this size. At 25
dimensions its matrix multiply is especially cheap, so on this set the graph
only catches up at larger corpora than the 128-dimension runs suggest.

Reproduce with the download command in `bench/public.py`, then
`python -m bench.public --n 20000`. The data file is 121 MB and gitignored.

### Two optimisations, one of which failed

Both measured rather than assumed, which is why one of them is not in the code.

**The one that failed.** Neighbour selection during insert compares each
candidate against everything already chosen. The obvious fix is to precompute
the whole candidate-to-candidate matrix in one matmul instead of one small call
per comparison. It made building 4,000 vectors **56% slower**, from 57 to 100
seconds. The loop usually stops after `capacity` acceptances, so most of that
matrix is computed and never read.

**The one that worked.** The same trick inside `_trim`, which re-runs the
heuristic on a neighbour that an insert pushed over its degree cap. There the
candidate list is barely larger than the capacity, so nearly all of the matrix
is read. That took building 4,000 vectors from 57 to **43 seconds**, a 24%
improvement, and it is in the code.

Same idea, opposite results, decided by measurement in both directions.

### Build cost

Roughly 90 vectors per second at `m=16, ef_construction=200`, so building
40,000 took 505 seconds. That is the honest cost of a graph build in Python:
each insert runs a full search of the layer, and each new edge may re-run the
selection heuristic on the node at the other end.

---

## Correctness

Recall reaching exactly 1.000 when `ef` covers the corpus is the headline check,
but a graph can score well and still be quietly broken. So the suite also
asserts structure directly:

| Property | Why it matters |
|---|---|
| Every node reachable from the entry point at layer zero | An unreachable node can never be returned however near it is |
| No node over its degree cap | `2m` on layer zero, `m` above |
| No self-loops, no duplicate edges | Both silently waste degree |
| Entry point sits on the top layer | Otherwise the descent starts in the wrong place |
| Layers thin out by roughly the branching factor | Confirms the exponential level draw |
| Edges are added both ways, and stay so at least 84% of the time | See below |
| Same seed builds the same graph | Makes every other test meaningful |

**On that 84%.** Every edge is created in both directions, but when an insert
pushes a neighbour over its cap, that neighbour re-runs the heuristic and may
drop the edge it was just handed while the new node keeps its side. The paper
prunes exactly this way. Measured here: 16% of edges end up one-way. The test
allows it and bounds it; reachability is the property that actually has to hold,
and it is checked separately.

Other behaviour under test: exact matches found under all three metrics,
duplicate vectors (where every distance is zero and a heuristic assuming strict
inequality can loop), NaN and infinity rejected at the door rather than
poisoning every later comparison, and a save-and-load round trip producing
byte-identical adjacency and identical query results.

```
$ .venv/bin/python -m pytest -q
.................................................................  [100%]
65 passed in 49.88s
```

---

## API

| Call | Does |
|---|---|
| `Index(dim, metric=, m=, ef_construction=, ef_search=, seed=)` | create |
| `add(id, vector)` | insert, or replace if the id exists |
| `add_many(ids, vectors)` | insert a 2-D array |
| `search(vector, k=10, ef=None)` | `Hit(id, distance)`, nearest first |
| `delete(id)` | hide an id; `False` if it was absent |
| `get(id)`, `ids()`, `len()`, `in` | inspect |
| `save(path)` / `Index.load(path)` | one `.npz` |
| `stats()` | nodes, deletions, layers, edges per layer |

`distance` is always **smaller is nearer**: cosine distance from 0 to 2,
euclidean distance, or the negated inner product. Inner product is not a metric
and nothing can make it one, so that last one can be negative.

### Two knobs

`ef_construction` decides how hard the build looks for good neighbours. `ef`
decides how hard a query looks. Raising either costs time and buys recall, and
`ef` is the only one you can change after the index exists.

### Deletion is soft

A deleted id is hidden from results, but its node stays in the graph. Removing
it would mean repairing every edge that pointed at it, and a graph that loses
waypoints loses the connectivity that makes search work. The cost is that
deleted vectors keep occupying memory and keep being traversed until the index
is rebuilt. Search widens itself in proportion to the deletions so a half-deleted
index still returns a full page of results, which is tested.

## Command line

```
sextant demo --n 5000 --dim 64          build a random index, print the curve
sextant build --vectors v.npy --out index.npz
sextant search --index index.npz --vector q.npy --k 10
sextant recall --index index.npz --vectors v.npy --queries q.npy
sextant stats --index index.npz
```

## Not built

- **Product quantisation.** Vectors are stored as float32. Compressing them is
  the next big memory win and a separate project.
- **A C extension.** The measurements above say this is where the time actually
  goes. It would move the crossover more than any algorithmic change left here.
- **Filtered search.** Restricting results by metadata during traversal, rather
  than filtering afterwards, changes the graph walk and is its own problem.
- **Incremental rebuild.** Deletions accumulate until you rebuild by hand.
- **Multi-threaded build.** Inserts take a graph-wide lock by construction here.

## Layout

```
sextant/
  distance.py  the three metrics, normalisation, and what gets reported
  hnsw.py      the graph: levels, insert, the heuristic, layered search
  index.py     ids, deletion, save and load, the recall helper
  brute.py     exact search, the reference every recall figure is measured against
  cli.py       the command line
bench/
  curve.py     recall and speed against ef, at one corpus size
  scaling.py   graph against exact search as the corpus grows, both data shapes
tests/         65 tests
```
