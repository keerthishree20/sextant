"""Distance metrics.

numpy does the arithmetic. It is the one dependency here, and it earns its place:
the graph, the layer assignment, the neighbour heuristic and the search are all
written out below, and none of that is what numpy would have done for us. What
it does is turn a dot product over a few hundred candidate vectors from a Python
loop into one call.

Every metric is expressed so that **smaller is nearer**. Inner product is
naturally the other way round, so it is negated, and the search never has to
know which metric it is running.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Metric = Literal["cosine", "l2", "inner_product"]
DType = np.float32


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length. Zero rows are left alone rather than
    producing a division by zero and a vector of NaN."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


def l2_squared(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Squared euclidean distance. The square root is monotonic, so leaving it
    out changes no ordering and saves the call."""
    diff = candidates - query
    return np.einsum("...i,...i->...", diff, diff)


def negative_inner_product(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return -(candidates @ query)


#: Cosine is inner product on unit vectors, so the index normalises on insert and
#: on query, and then reuses exactly the same code path.
_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cosine": negative_inner_product,
    "inner_product": negative_inner_product,
    "l2": l2_squared,
}


def pairwise(metric: Metric, vectors: np.ndarray) -> np.ndarray:
    """Every ordering value between every pair, in one matrix multiply.

    Worth it only when nearly all of the matrix gets read. See `HNSW._trim`,
    which does, and the note in `HNSW._select_neighbours`, which does not.
    """
    if metric in ("cosine", "inner_product"):
        return -(vectors @ vectors.T)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    return np.maximum(squared[:, None] + squared[None, :] - 2.0 * (vectors @ vectors.T), 0.0)


def function_for(metric: Metric) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        return _FUNCTIONS[metric]
    except KeyError:
        raise ValueError(
            f"unknown metric {metric!r}; choose one of {', '.join(sorted(_FUNCTIONS))}"
        ) from None


def normalises(metric: Metric) -> bool:
    """Whether vectors should be scaled to unit length before storage."""
    return metric == "cosine"


def report(metric: Metric, ordering_value: float) -> float:
    """Turn the internal ordering value into the number a caller sees.

    Every metric reports something where **smaller is nearer**, so results read
    the same way whichever one is in use:

    * ``cosine`` gives cosine distance, ``1 - similarity``, from 0 to 2.
    * ``l2`` gives the euclidean distance, with the square root the search
      itself has no reason to compute.
    * ``inner_product`` gives the negated inner product. That preserves the
      ordering but is not a true distance, and it can be negative. Inner
      product is not a metric; nothing can make it one.
    """
    if metric == "cosine":
        return float(1.0 + ordering_value)
    if metric == "l2":
        return float(np.sqrt(max(ordering_value, 0.0)))
    return float(ordering_value)
