"""Hierarchical Navigable Small World graph.

The idea in one paragraph. Build a proximity graph where every node links to its
nearest neighbours, and searching becomes greedy walking: step to whichever
neighbour is closer to the query, stop when none is. That works, but a greedy
walk on a flat graph takes a long time to cross the space. So stack the graph
into layers, each a sparser sample of the one below, and start at the top. The
upper layers move you across the space in a few long strides; the bottom layer
does the fine approach. The layer count for each node is drawn from an
exponential distribution, which is what makes the upper layers sparse without
anyone choosing who goes in them.

Written from the algorithms in Malkov and Yashunin, "Efficient and robust
approximate nearest neighbor search using Hierarchical Navigable Small World
graphs" (2016). The neighbour selection below is their Algorithm 4, the part
that matters most and the part a naive implementation leaves out.
"""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from . import distance
from .distance import DType, Metric

INITIAL_CAPACITY = 1024


class HNSW:
    def __init__(
        self,
        dim: int,
        *,
        metric: Metric = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        seed: int | None = None,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if m < 2:
            raise ValueError("m must be at least 2; below that the graph stops connecting")
        if ef_construction < m:
            raise ValueError("ef_construction must be at least m")

        self.dim = dim
        self.metric: Metric = metric
        self.m = m
        #: Layer zero carries twice the degree. It is the layer every search
        #: finishes on, so its connectivity decides recall.
        self.m0 = 2 * m
        self.ef_construction = ef_construction

        self._distance = distance.function_for(metric)
        self._normalise = distance.normalises(metric)
        self._rng = np.random.default_rng(seed)
        #: The level distribution's scale. 1/ln(m) makes each layer about 1/m
        #: the size of the one below it.
        self._level_scale = 1.0 / math.log(m)

        self._vectors = np.zeros((INITIAL_CAPACITY, dim), dtype=DType)
        self._count = 0
        self._levels: list[int] = []
        #: One adjacency map per layer: node -> its neighbours on that layer.
        self._links: list[dict[int, list[int]]] = []
        self._entry: int | None = None
        self._max_level = -1

    # ------------------------------------------------------------------ shape

    def __len__(self) -> int:
        return self._count

    @property
    def entry_point(self) -> int | None:
        return self._entry

    @property
    def max_level(self) -> int:
        return self._max_level

    def levels(self) -> list[int]:
        return list(self._levels)

    def neighbours(self, node: int, level: int) -> list[int]:
        return list(self._links[level].get(node, ())) if level < len(self._links) else []

    def vector(self, node: int) -> np.ndarray:
        return self._vectors[node]

    # ----------------------------------------------------------------- insert

    def add(self, vector: np.ndarray) -> int:
        """Insert one vector. Returns its internal node number."""
        vector = self._prepare(vector)
        node = self._count
        self._reserve(node + 1)
        self._vectors[node] = vector
        self._count += 1

        level = self._draw_level()
        self._levels.append(level)
        while len(self._links) <= level:
            self._links.append({})
        for lc in range(level + 1):
            self._links[lc].setdefault(node, [])

        if self._entry is None:
            self._entry = node
            self._max_level = level
            return node

        cursor = self._entry
        # Above the new node's own top layer there is nothing to link, so just
        # walk downhill to get a good entry point for the layers that matter.
        for lc in range(self._max_level, level, -1):
            cursor = self._descend(vector, cursor, lc)

        for lc in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(vector, [cursor], self.ef_construction, lc)
            capacity = self.m0 if lc == 0 else self.m
            chosen = self._select_neighbours(vector, found, capacity)

            self._links[lc][node] = list(chosen)
            for other in chosen:
                # Links are two-way. A one-way edge is invisible to any search
                # arriving from the other side, which is how a graph ends up
                # with unreachable islands.
                self._links[lc].setdefault(other, []).append(node)
                self._trim(other, lc, capacity)

            if found:
                cursor = found[0][1]

        if level > self._max_level:
            self._entry = node
            self._max_level = level
        return node

    def _draw_level(self) -> int:
        """An exponential draw. Most nodes land on layer zero and only a handful
        ever reach the top, which is what makes the upper layers a usable index
        rather than a copy of the data."""
        return int(-math.log(max(self._rng.random(), 1e-12)) * self._level_scale)

    def _trim(self, node: int, level: int, capacity: int) -> None:
        """Re-run the heuristic on a node that an insert pushed over its cap.

        This is the hot path of the whole build: it runs once per new edge, so
        tens of times per insert. Unlike the selection during insert, it looks
        at a candidate list barely larger than the capacity, so it reads almost
        the entire candidate-to-candidate matrix. That is what makes computing
        the matrix up front pay here and lose there.
        """
        links = self._links[level][node]
        if len(links) <= capacity:
            return
        vector = self._vectors[node]
        neighbours = self._vectors[links]
        distances = self._distance(vector, neighbours)
        order = np.argsort(distances)
        ordered = [links[i] for i in order]
        between = distance.pairwise(self.metric, neighbours[order])

        chosen_at: list[int] = []
        discarded: list[int] = []
        for position in range(len(ordered)):
            if len(chosen_at) >= capacity:
                break
            if not chosen_at or float(distances[order[position]]) < float(
                    np.min(between[position, chosen_at])):
                chosen_at.append(position)
            else:
                discarded.append(position)
        for position in discarded:
            if len(chosen_at) >= capacity:
                break
            chosen_at.append(position)

        self._links[level][node] = [ordered[position] for position in chosen_at]

    # ----------------------------------------------------------------- search

    def search(self, vector: np.ndarray, k: int, *, ef: int | None = None,
               skip: Iterable[int] = ()) -> list[tuple[float, int]]:
        """The `ef` nearest candidates are explored; the best `k` come back.

        Raising `ef` trades query time for recall, and it is the only knob that
        matters at query time. It is clamped up to `k`, since asking for ten
        results while exploring five cannot answer the question.
        """
        if self._entry is None or k <= 0:
            return []
        ef = max(ef if ef is not None else max(k, 32), k)
        vector = self._prepare(vector)

        cursor = self._entry
        for lc in range(self._max_level, 0, -1):
            cursor = self._descend(vector, cursor, lc)

        found = self._search_layer(vector, [cursor], ef, 0)
        skip = set(skip)
        if skip:
            found = [pair for pair in found if pair[1] not in skip]
        return found[:k]

    def _descend(self, vector: np.ndarray, cursor: int, level: int) -> int:
        """Greedy walk on one layer: keep stepping to a closer neighbour."""
        best = float(self._distance(vector, self._vectors[cursor:cursor + 1])[0])
        improved = True
        while improved:
            improved = False
            links = self._links[level].get(cursor)
            if not links:
                break
            distances = self._distance(vector, self._vectors[links])
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) < best:
                best = float(distances[nearest])
                cursor = links[nearest]
                improved = True
        return cursor

    def _search_layer(self, vector: np.ndarray, entries: list[int], ef: int,
                      level: int) -> list[tuple[float, int]]:
        """Best-first search on one layer, keeping the `ef` closest seen.

        Two heaps: `candidates` is a min-heap of places still worth visiting,
        `results` is a max-heap of the best found so far. The search stops when
        the nearest unvisited candidate is further away than the worst result
        already held, because nothing beyond it can improve the set.
        """
        visited = set(entries)
        starts = self._distance(vector, self._vectors[entries])

        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        for node, dist in zip(entries, starts):
            heapq.heappush(candidates, (float(dist), node))
            heapq.heappush(results, (-float(dist), node))

        links = self._links[level]
        while candidates:
            nearest_dist, nearest = heapq.heappop(candidates)
            if -results[0][0] < nearest_dist and len(results) >= ef:
                break

            unvisited = [n for n in links.get(nearest, ()) if n not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)

            # One numpy call for the whole neighbourhood rather than one per
            # edge. This is where the time goes, so it is the loop worth
            # keeping out of Python.
            for node, dist in zip(unvisited, self._distance(vector, self._vectors[unvisited])):
                dist = float(dist)
                if len(results) < ef:
                    heapq.heappush(candidates, (dist, node))
                    heapq.heappush(results, (-dist, node))
                elif dist < -results[0][0]:
                    heapq.heappush(candidates, (dist, node))
                    heapq.heapreplace(results, (-dist, node))

        return sorted((-d, n) for d, n in results)

    # ------------------------------------------------------- neighbour choice

    def _select_neighbours(self, vector: np.ndarray, candidates: list[tuple[float, int]],
                           capacity: int) -> list[int]:
        """Algorithm 4: pick close neighbours that are not close to each other.

        Taking the `capacity` nearest candidates is the obvious approach and it
        builds a bad graph. In a dense cluster every one of those links points
        the same way, the node has no edge leading out of the cluster, and a
        greedy search that arrives there cannot leave. The rule below only
        accepts a candidate that is closer to the new node than to anything
        already accepted, which forces the links to spread out and keeps the
        graph navigable.
        """
        if len(candidates) <= capacity:
            return [node for _dist, node in candidates]

        chosen: list[int] = []
        discarded: list[tuple[float, int]] = []
        for dist, node in sorted(candidates):
            if len(chosen) >= capacity:
                break
            if not chosen:
                chosen.append(node)
                continue
            # Measured, not assumed: precomputing the full candidate-to-candidate
            # matrix in one matmul is the obvious optimisation and it made
            # building 4,000 vectors 56% slower. The loop below usually stops
            # after `capacity` acceptances, so most of that matrix is never read.
            to_chosen = self._distance(self._vectors[node], self._vectors[chosen])
            if dist < float(np.min(to_chosen)):
                chosen.append(node)
            else:
                discarded.append((dist, node))

        # Rather than leave the node under-connected, fill the remaining slots
        # with the nearest of the rejected candidates.
        for _dist, node in discarded:
            if len(chosen) >= capacity:
                break
            chosen.append(node)
        return chosen

    # -------------------------------------------------------------- plumbing

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=DType)
        if array.shape != (self.dim,):
            raise ValueError(
                f"expected a vector of {self.dim} dimensions, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("vector contains NaN or infinity")
        return distance.normalise(array) if self._normalise else array

    def _reserve(self, needed: int) -> None:
        if needed <= len(self._vectors):
            return
        capacity = len(self._vectors)
        while capacity < needed:
            capacity *= 2
        grown = np.zeros((capacity, self.dim), dtype=DType)
        grown[:self._count] = self._vectors[:self._count]
        self._vectors = grown

    def raw_vectors(self) -> np.ndarray:
        """The stored vectors, normalised if the metric called for it."""
        return self._vectors[:self._count]

    def link_counts(self) -> list[int]:
        return [sum(len(v) for v in layer.values()) for layer in self._links]
