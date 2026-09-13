"""Recall against exact search.

An approximate index is only worth anything if you know how approximate. Query
speed on its own says nothing: returning the wrong answers is always fast.
Everything here is measured against `BruteForce`, which compares the query to
every vector and cannot be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from sextant import BruteForce, Index, build, recall_at_k

N = 800
DIM = 32


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(2026)
    data = rng.normal(size=(N, DIM)).astype(np.float32)
    queries = rng.normal(size=(60, DIM)).astype(np.float32)
    return data, queries


@pytest.fixture(scope="module")
def pair(dataset):
    data, _queries = dataset
    ids = [f"v{i}" for i in range(N)]
    index = build(ids, data, metric="cosine", m=12, ef_construction=100, seed=8)
    exact = BruteForce(DIM, metric="cosine")
    exact.add_many(ids, data)
    return index, exact


def _recall(index: Index, exact: BruteForce, queries, k: int, ef: int) -> float:
    got = [[hit.id for hit in index.search(q, k, ef=ef)] for q in queries]
    truth = [exact.search_ids(q, k) for q in queries]
    return recall_at_k(got, truth, k)


def test_a_wide_search_finds_everything_exact_search_finds(pair, dataset):
    """With ef at the size of the dataset the graph has nowhere left to miss."""
    index, exact = pair
    _data, queries = dataset
    assert _recall(index, exact, queries, 10, ef=N) == pytest.approx(1.0)


def test_recall_climbs_with_ef(pair, dataset):
    """The single trade-off this index offers, and the curve has to be
    monotonic or the knob is not a knob."""
    index, exact = pair
    _data, queries = dataset
    curve = [_recall(index, exact, queries, 10, ef=ef) for ef in (10, 20, 40, 80, 160)]
    assert curve == sorted(curve), f"recall went backwards: {curve}"
    assert curve[0] < curve[-1], "ef made no difference at all"
    assert curve[-1] > 0.95, f"recall at ef=160 was only {curve[-1]:.3f}"


def test_the_default_settings_are_usable_out_of_the_box(pair, dataset):
    """Whatever the tuning story, an index nobody configures has to be good."""
    index, exact = pair
    _data, queries = dataset
    got = [[hit.id for hit in index.search(q, 10)] for q in queries]
    truth = [exact.search_ids(q, 10) for q in queries]
    assert recall_at_k(got, truth, 10) > 0.90


def test_recall_at_one_is_higher_than_recall_at_ten(pair, dataset):
    """The nearest neighbour is the easiest one to find; a graph that loses it
    is broken, not approximate."""
    index, exact = pair
    _data, queries = dataset
    assert _recall(index, exact, queries, 1, ef=64) >= _recall(index, exact, queries, 10, ef=64)


def test_reported_distances_match_exact_search(pair, dataset):
    index, exact = pair
    _data, queries = dataset
    for query in queries[:10]:
        approximate = index.search(query, 5, ef=N)
        truth = exact.search(query, 5)
        for hit, (id, score) in zip(approximate, truth):
            assert hit.id == id
            assert hit.distance == pytest.approx(score, abs=1e-4)


@pytest.mark.parametrize("metric", ["l2", "inner_product"])
def test_the_other_metrics_reach_full_recall_too(dataset, metric):
    data, queries = dataset
    ids = [f"v{i}" for i in range(N)]
    index = build(ids, data, metric=metric, m=12, ef_construction=100, seed=8)
    exact = BruteForce(DIM, metric=metric)
    exact.add_many(ids, data)
    assert _recall(index, exact, queries[:20], 10, ef=N) == pytest.approx(1.0)


def test_clustered_data_does_not_trap_the_search():
    """The case the neighbour heuristic exists for. Tight clusters far apart:
    a graph that links every node only to its nearest few has no edge leading
    out of a cluster, and a greedy search that lands in the wrong one is stuck.
    """
    rng = np.random.default_rng(4)
    centres = rng.normal(size=(6, 24)).astype(np.float32) * 12
    data = np.vstack([centre + rng.normal(size=(90, 24)).astype(np.float32) * 0.4
                      for centre in centres])
    ids = [f"v{i}" for i in range(len(data))]

    index = build(ids, data, metric="l2", m=12, ef_construction=100, seed=9)
    exact = BruteForce(24, metric="l2")
    exact.add_many(ids, data)

    queries = data[rng.choice(len(data), 40, replace=False)] + \
        rng.normal(size=(40, 24)).astype(np.float32) * 0.1
    got = [[hit.id for hit in index.search(q, 10, ef=100)] for q in queries]
    truth = [exact.search_ids(q, 10) for q in queries]
    assert recall_at_k(got, truth, 10) > 0.95


def test_duplicate_vectors_do_not_break_the_graph():
    """Every distance is zero, so nothing distinguishes the candidates. A
    heuristic that assumes strict inequality can loop or return nothing here."""
    data = np.tile(np.eye(1, 8, dtype=np.float32), (60, 1))
    index = build([f"v{i}" for i in range(60)], data, m=8, ef_construction=64, seed=1)
    hits = index.search(data[0], k=10, ef=60)
    assert len(hits) == 10
    assert all(hit.distance == pytest.approx(0.0, abs=1e-5) for hit in hits)


def test_recall_helper_counts_what_it_says():
    assert recall_at_k([["a", "b", "c"]], [["a", "b", "c"]], 3) == 1.0
    assert recall_at_k([["a", "x", "y"]], [["a", "b", "c"]], 3) == pytest.approx(1 / 3)
    assert recall_at_k([["x"]], [["a"]], 1) == 0.0
    assert recall_at_k([], [], 10) == 0.0
