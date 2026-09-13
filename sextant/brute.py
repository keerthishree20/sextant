"""Exact search, for measuring the approximate one against.

Every recall figure in this project is computed against this file. It has no
index and no cleverness: it compares the query to everything. That is what makes
it trustworthy as a reference, and it is also what makes it unusable past a
certain size, which is the entire reason the graph exists.
"""

from __future__ import annotations

import numpy as np

from . import distance
from .distance import DType, Metric


class BruteForce:
    def __init__(self, dim: int, *, metric: Metric = "cosine"):
        self.dim = dim
        self.metric: Metric = metric
        self._distance = distance.function_for(metric)
        self._normalise = distance.normalises(metric)
        self._ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=DType)

    def __len__(self) -> int:
        return len(self._ids)

    def add_many(self, ids, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=DType)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} dimensions, got {vectors.shape[1]}")
        if self._normalise:
            vectors = distance.normalise(vectors)
        self._ids.extend(ids)
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        query = np.asarray(vector, dtype=DType)
        if self._normalise:
            query = distance.normalise(query)
        distances = self._distance(query, self._vectors)
        k = min(k, len(self._ids))
        # argpartition finds the k smallest without sorting the rest, then only
        # those k are sorted. On a million vectors that is the difference
        # between a full sort and a linear pass.
        candidates = np.argpartition(distances, k - 1)[:k]
        ordered = candidates[np.argsort(distances[candidates])]
        return [(self._ids[i], distance.report(self.metric, float(distances[i])))
                for i in ordered]

    def search_ids(self, vector: np.ndarray, k: int = 10) -> list[str]:
        return [id for id, _score in self.search(vector, k)]
