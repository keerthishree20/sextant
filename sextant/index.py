"""The index a caller actually uses: string ids, deletions, and a file on disk.

`HNSW` speaks in node numbers and ordering values. This wraps it in the things
an application needs and keeps that bookkeeping out of the graph.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

from . import distance
from .distance import DType, Metric
from .hnsw import HNSW

FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Hit:
    id: str
    #: Smaller is nearer, for every metric. See `distance.report`.
    distance: float


class Index:
    def __init__(
        self,
        dim: int,
        *,
        metric: Metric = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int | None = None,
    ):
        self.graph = HNSW(dim, metric=metric, m=m, ef_construction=ef_construction, seed=seed)
        self.ef_search = ef_search
        self._ids: list[str] = []            # node number -> id
        self._nodes: dict[str, int] = {}     # id -> node number
        self._deleted: set[int] = set()

    # -------------------------------------------------------------- properties

    @property
    def dim(self) -> int:
        return self.graph.dim

    @property
    def metric(self) -> Metric:
        return self.graph.metric

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, id: str) -> bool:
        return id in self._nodes

    def ids(self) -> Iterator[str]:
        return iter(list(self._nodes))

    # ------------------------------------------------------------------- write

    def add(self, id: str, vector: np.ndarray) -> None:
        """Insert one vector under an id. Re-adding an id replaces it."""
        if not isinstance(id, str):
            raise TypeError(f"ids are strings, got {type(id).__name__}")
        if id in self._nodes:
            # Replacement is a delete plus an insert. The old node stays in the
            # graph as a waypoint, which is the same bargain every deletion
            # makes here.
            self.delete(id)
        node = self.graph.add(vector)
        self._ids.append(id)
        self._nodes[id] = node

    def add_many(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=DType)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} ids but {len(vectors)} vectors")
        for id, vector in zip(ids, vectors):
            self.add(id, vector)

    def delete(self, id: str) -> bool:
        """Hide an id from results. False means it was not there.

        The node stays in the graph. Removing it would mean repairing every
        edge that pointed at it, and a graph that loses waypoints loses the
        connectivity that makes the search work. The cost is that deleted
        vectors keep occupying memory and keep being traversed until the index
        is rebuilt.
        """
        node = self._nodes.pop(id, None)
        if node is None:
            return False
        self._deleted.add(node)
        return True

    # -------------------------------------------------------------------- read

    def search(self, vector: np.ndarray, k: int = 10, *, ef: int | None = None) -> list[Hit]:
        if k <= 0:
            return []
        ef = ef if ef is not None else self.ef_search
        ef = max(ef, k)

        if self._deleted:
            # Deleted nodes are still traversed and still occupy result slots,
            # so widen the search in proportion to how many there are. Without
            # this, an index that is half deleted quietly returns half as many
            # results as it was asked for.
            live = max(len(self._nodes), 1)
            total = live + len(self._deleted)
            ef = min(int(ef * total / live) + k, total)

        found = self.graph.search(vector, k, ef=ef, skip=self._deleted)
        return [Hit(self._ids[node], distance.report(self.metric, dist))
                for dist, node in found[:k]]

    def get(self, id: str) -> np.ndarray | None:
        node = self._nodes.get(id)
        return None if node is None else self.graph.vector(node).copy()

    # ------------------------------------------------------------------- state

    def stats(self) -> dict[str, object]:
        counts = self.graph.link_counts()
        return {
            "ids": len(self._nodes),
            "nodes": len(self.graph),
            "deleted": len(self._deleted),
            "dim": self.dim,
            "metric": self.metric,
            "m": self.graph.m,
            "ef_construction": self.graph.ef_construction,
            "ef_search": self.ef_search,
            "layers": self.graph.max_level + 1,
            "edges_per_layer": counts,
            "edges": sum(counts),
        }

    # ---------------------------------------------------------------- on disk

    def save(self, path: str | pathlib.Path) -> None:
        """One `.npz`. The adjacency lists go in as flat arrays with offsets
        rather than as JSON, which keeps a large graph a fraction of the size
        and loads without parsing anything."""
        arrays: dict[str, np.ndarray] = {
            "vectors": self.graph.raw_vectors(),
            "levels": np.asarray(self.graph.levels(), dtype=np.int32),
        }
        for level in range(self.graph.max_level + 1):
            nodes = sorted(n for n in range(len(self.graph))
                           if self.graph.neighbours(n, level) or self.graph.levels()[n] >= level)
            flat: list[int] = []
            offsets = [0]
            for node in nodes:
                flat.extend(self.graph.neighbours(node, level))
                offsets.append(len(flat))
            arrays[f"nodes_{level}"] = np.asarray(nodes, dtype=np.int32)
            arrays[f"offsets_{level}"] = np.asarray(offsets, dtype=np.int32)
            arrays[f"flat_{level}"] = np.asarray(flat, dtype=np.int32)

        meta = {
            "version": FORMAT_VERSION,
            "dim": self.dim,
            "metric": self.metric,
            "m": self.graph.m,
            "ef_construction": self.graph.ef_construction,
            "ef_search": self.ef_search,
            "entry": self.graph.entry_point,
            "max_level": self.graph.max_level,
            "ids": self._ids,
            "deleted": sorted(self._deleted),
        }
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.with_suffix(target.suffix + ".tmp")
        with open(scratch, "wb") as handle:
            np.savez(handle, **arrays)
        scratch.replace(target)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Index":
        with np.load(pathlib.Path(path), allow_pickle=False) as bundle:
            meta = json.loads(bytes(bundle["meta"]).decode())
            if meta["version"] != FORMAT_VERSION:
                raise ValueError(
                    f"index was written by format version {meta['version']}, "
                    f"this build reads version {FORMAT_VERSION}")

            index = cls(meta["dim"], metric=meta["metric"], m=meta["m"],
                        ef_construction=meta["ef_construction"],
                        ef_search=meta["ef_search"])
            graph = index.graph
            vectors = bundle["vectors"]
            graph._vectors = np.ascontiguousarray(vectors, dtype=DType)
            graph._count = len(vectors)
            graph._levels = [int(v) for v in bundle["levels"]]
            graph._entry = meta["entry"]
            graph._max_level = meta["max_level"]
            graph._links = []

            for level in range(meta["max_level"] + 1):
                nodes = bundle[f"nodes_{level}"]
                offsets = bundle[f"offsets_{level}"]
                flat = bundle[f"flat_{level}"]
                layer: dict[int, list[int]] = {}
                for position, node in enumerate(nodes):
                    start, end = int(offsets[position]), int(offsets[position + 1])
                    layer[int(node)] = [int(n) for n in flat[start:end]]
                graph._links.append(layer)

            index._ids = list(meta["ids"])
            index._deleted = set(meta["deleted"])
            index._nodes = {id: node for node, id in enumerate(index._ids)
                            if node not in index._deleted}
        return index

    def __repr__(self) -> str:
        return (f"<Index {len(self)} ids, {len(self.graph)} nodes, "
                f"dim={self.dim}, metric={self.metric}>")


def build(ids: Sequence[str], vectors: np.ndarray, **kwargs) -> Index:
    """Convenience: an index containing exactly these vectors."""
    vectors = np.asarray(vectors, dtype=DType)
    index = Index(vectors.shape[1], **kwargs)
    index.add_many(ids, vectors)
    return index


def recall_at_k(approximate: Iterable[Sequence[str]],
                exact: Iterable[Sequence[str]], k: int) -> float:
    """The fraction of true nearest neighbours the index actually returned.

    The only measure of an approximate index that means anything. Query speed
    without a recall figure beside it says nothing at all: returning the wrong
    answers is always fast.
    """
    total = 0
    hits = 0
    for got, want in zip(approximate, exact):
        truth = set(want[:k])
        hits += len(truth.intersection(got[:k]))
        total += len(truth)
    return hits / total if total else 0.0
