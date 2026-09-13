"""Sextant: an HNSW vector index with the graph written from scratch.

    import numpy as np
    from sextant import Index

    index = Index(dim=128, metric="cosine")
    index.add("doc-1", np.random.rand(128))
    index.search(query, k=10)
"""

from .brute import BruteForce
from .distance import Metric
from .hnsw import HNSW
from .index import Hit, Index, build, recall_at_k

__version__ = "1.0.0"
__all__ = ["Index", "Hit", "HNSW", "BruteForce", "Metric", "build", "recall_at_k"]
