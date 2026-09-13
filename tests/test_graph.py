"""Structural properties of the graph itself.

Recall tests say the index works. These say why. A graph can score acceptably on
a small set while carrying one-way edges or an unreachable pocket, and those
faults only show up later, on data nobody tests with.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from sextant.hnsw import HNSW


@pytest.fixture
def graph() -> HNSW:
    rng = np.random.default_rng(11)
    index = HNSW(24, metric="cosine", m=8, ef_construction=100, seed=3)
    for vector in rng.normal(size=(600, 24)).astype(np.float32):
        index.add(vector)
    return index


def test_every_vector_becomes_a_node(graph):
    assert len(graph) == 600
    assert len(graph.levels()) == 600


def test_edges_are_created_both_ways(graph):
    """Every edge is added in both directions. Most stay that way.

    Some do not, and that is the algorithm rather than a bug. When an insert
    pushes a neighbour over its degree cap, the neighbour re-runs the selection
    heuristic and may drop the edge it was just given, while the new node keeps
    its side. The paper's Algorithm 1 prunes exactly this way.

    Asymmetry is therefore allowed but must stay rare: it is a side effect of
    pruning a full node, not the normal state of an edge. The property that
    actually has to hold is reachability, which the next test checks.
    """
    one_way = 0
    total = 0
    for level in range(graph.max_level + 1):
        for node in range(len(graph)):
            for neighbour in graph.neighbours(node, level):
                total += 1
                if node not in graph.neighbours(neighbour, level):
                    one_way += 1
    assert total > 0
    assert one_way / total < 0.25, f"{one_way} of {total} edges point only one way"


def test_no_node_exceeds_its_degree_cap(graph):
    for level in range(graph.max_level + 1):
        cap = graph.m0 if level == 0 else graph.m
        for node in range(len(graph)):
            assert len(graph.neighbours(node, level)) <= cap


def test_no_node_links_to_itself(graph):
    for level in range(graph.max_level + 1):
        for node in range(len(graph)):
            assert node not in graph.neighbours(node, level)


def test_no_edge_is_listed_twice(graph):
    for level in range(graph.max_level + 1):
        for node in range(len(graph)):
            links = graph.neighbours(node, level)
            assert len(links) == len(set(links)), f"duplicate edge on node {node}"


def test_a_node_only_appears_on_layers_it_reaches(graph):
    levels = graph.levels()
    for level in range(graph.max_level + 1):
        for node in range(len(graph)):
            if graph.neighbours(node, level):
                assert levels[node] >= level


def test_the_entry_point_sits_on_the_top_layer(graph):
    assert graph.entry_point is not None
    assert graph.levels()[graph.entry_point] == graph.max_level


def test_every_node_is_reachable_from_the_entry_point(graph):
    """An unreachable node can never be returned, however close it is to the
    query. Layer zero holds every node, so this is the layer that has to be
    connected."""
    seen = {graph.entry_point}
    queue = collections.deque([graph.entry_point])
    while queue:
        node = queue.popleft()
        for neighbour in graph.neighbours(node, 0):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    assert len(seen) == len(graph), f"{len(graph) - len(seen)} nodes are cut off"


def test_layers_thin_out_by_roughly_the_branching_factor(graph):
    """The level draw is exponential with scale 1/ln(m), so each layer should
    hold about 1/m of the one below. Loose bounds: this is a random draw."""
    counts = collections.Counter(graph.levels())
    total = len(graph)
    at_least_one = sum(count for level, count in counts.items() if level >= 1)
    assert 0.02 < at_least_one / total < 0.30
    assert counts[0] > total * 0.7


def test_the_top_layer_is_small(graph):
    counts = collections.Counter(graph.levels())
    assert counts[graph.max_level] <= 3


def test_the_same_seed_builds_the_same_graph():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(200, 16)).astype(np.float32)

    def build(seed: int) -> list[int]:
        graph = HNSW(16, m=8, ef_construction=64, seed=seed)
        for vector in data:
            graph.add(vector)
        return graph.levels()

    assert build(42) == build(42)
    assert build(42) != build(43)


def test_a_single_vector_is_its_own_entry_point():
    graph = HNSW(4, seed=1)
    node = graph.add(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert graph.entry_point == node
    assert graph.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 5) == [
        (pytest.approx(-1.0, abs=1e-5), node)]


def test_an_empty_graph_searches_to_nothing():
    graph = HNSW(4, seed=1)
    assert graph.search(np.zeros(4, dtype=np.float32), 5) == []


def test_a_wrongly_shaped_vector_is_refused():
    graph = HNSW(4, seed=1)
    with pytest.raises(ValueError, match="4 dimensions"):
        graph.add(np.zeros(5, dtype=np.float32))


def test_a_vector_with_nan_is_refused_at_the_door():
    """One NaN in the store makes every comparison against it false, and the
    node silently stops being findable. Better to reject it here."""
    graph = HNSW(3, seed=1)
    with pytest.raises(ValueError, match="NaN or infinity"):
        graph.add(np.array([1.0, np.nan, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinity"):
        graph.add(np.array([1.0, np.inf, 0.0], dtype=np.float32))


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="dim must be positive"):
        HNSW(0)
    with pytest.raises(ValueError, match="m must be at least 2"):
        HNSW(4, m=1)
    with pytest.raises(ValueError, match="ef_construction must be at least m"):
        HNSW(4, m=16, ef_construction=8)


def test_capacity_grows_past_the_initial_block():
    from sextant.hnsw import INITIAL_CAPACITY

    rng = np.random.default_rng(2)
    graph = HNSW(8, m=4, ef_construction=32, seed=1)
    for vector in rng.normal(size=(INITIAL_CAPACITY + 50, 8)).astype(np.float32):
        graph.add(vector)
    assert len(graph) == INITIAL_CAPACITY + 50
    assert graph.raw_vectors().shape[0] == INITIAL_CAPACITY + 50
