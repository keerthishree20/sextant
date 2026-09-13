"""The index as an API: ids, replacement, deletion, and the edges."""

from __future__ import annotations

import numpy as np
import pytest

from sextant import Index, build


@pytest.fixture
def vectors() -> np.ndarray:
    return np.random.default_rng(19).normal(size=(200, 16)).astype(np.float32)


@pytest.fixture
def index(vectors) -> Index:
    return build([f"v{i}" for i in range(len(vectors))], vectors,
                 metric="cosine", m=8, ef_construction=64, seed=4)


def test_a_vector_finds_itself_first(index, vectors):
    hits = index.search(vectors[42], k=3)
    assert hits[0].id == "v42"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-4)


def test_results_come_back_nearest_first(index, vectors):
    hits = index.search(vectors[7], k=10)
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_ids_and_membership(index):
    assert len(index) == 200
    assert "v0" in index
    assert "missing" not in index
    assert set(index.ids()) == {f"v{i}" for i in range(200)}


def test_get_returns_the_stored_vector(index, vectors):
    stored = index.get("v3")
    assert stored is not None
    # Cosine normalises on the way in, so direction is preserved, not length.
    direction = vectors[3] / np.linalg.norm(vectors[3])
    np.testing.assert_allclose(stored, direction, atol=1e-5)
    assert index.get("nope") is None


def test_asking_for_more_than_there_is_returns_what_there_is(vectors):
    small = build(["a", "b", "c"], vectors[:3], m=8, ef_construction=64, seed=1)
    assert len(small.search(vectors[0], k=50)) == 3


def test_an_empty_index_searches_to_nothing():
    empty = Index(8, seed=1)
    assert empty.search(np.zeros(8, dtype=np.float32), k=5) == []
    assert len(empty) == 0


def test_asking_for_zero_results_returns_none(index, vectors):
    assert index.search(vectors[0], k=0) == []


def test_a_deleted_id_never_comes_back(index, vectors):
    assert index.delete("v42") is True
    assert "v42" not in index
    assert len(index) == 199
    for hit in index.search(vectors[42], k=20):
        assert hit.id != "v42"


def test_deleting_twice_reports_the_second_as_absent(index):
    assert index.delete("v1") is True
    assert index.delete("v1") is False


def test_deletion_still_returns_a_full_page_of_results(index, vectors):
    """A half-deleted index must not quietly return half as many results. The
    search widens itself to make up for the nodes it has to skip."""
    for i in range(0, 200, 2):
        index.delete(f"v{i}")
    assert len(index) == 100

    hits = index.search(vectors[1], k=10)
    assert len(hits) == 10
    assert all(int(h.id[1:]) % 2 == 1 for h in hits)


def test_deleting_everything_leaves_an_index_that_answers_nothing(index, vectors):
    for i in range(200):
        index.delete(f"v{i}")
    assert len(index) == 0
    assert index.search(vectors[0], k=5) == []


def test_re_adding_an_id_replaces_its_vector(index):
    replacement = np.zeros(16, dtype=np.float32)
    replacement[0] = 1.0
    index.add("v5", replacement)

    assert len(index) == 200, "replacement should not change the id count"
    np.testing.assert_allclose(index.get("v5"), replacement, atol=1e-6)
    hits = index.search(replacement, k=1, ef=200)
    assert hits[0].id == "v5"


def test_ids_must_be_strings(index):
    with pytest.raises(TypeError, match="ids are strings"):
        index.add(7, np.zeros(16, dtype=np.float32))


def test_a_wrongly_shaped_vector_is_refused(index):
    with pytest.raises(ValueError, match="16 dimensions"):
        index.add("bad", np.zeros(4, dtype=np.float32))


def test_add_many_checks_the_lengths_line_up():
    index = Index(4, seed=1)
    with pytest.raises(ValueError, match="2 ids but 3 vectors"):
        index.add_many(["a", "b"], np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D array"):
        index.add_many(["a"], np.zeros(4, dtype=np.float32))


def test_stats_describe_the_graph(index):
    stats = index.stats()
    assert stats["ids"] == 200
    assert stats["nodes"] == 200
    assert stats["deleted"] == 0
    assert stats["metric"] == "cosine"
    assert stats["layers"] >= 1
    assert stats["edges"] > 0
    assert len(stats["edges_per_layer"]) == stats["layers"]


def test_deleted_nodes_stay_in_the_graph(index):
    """Stated as a test because it is the cost of soft deletion: the vector is
    hidden, not reclaimed."""
    index.delete("v9")
    stats = index.stats()
    assert stats["ids"] == 199
    assert stats["nodes"] == 200
    assert stats["deleted"] == 1


@pytest.mark.parametrize("metric", ["cosine", "l2", "inner_product"])
def test_every_metric_finds_an_exact_match(vectors, metric):
    index = build([f"v{i}" for i in range(50)], vectors[:50],
                  metric=metric, m=8, ef_construction=64, seed=2)
    hits = index.search(vectors[3], k=1, ef=50)
    assert hits[0].id == "v3"
