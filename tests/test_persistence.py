"""Saving and loading.

A loaded index has to be the same index, not a similar one. Identical results
for identical queries is the only assertion that catches a graph that came back
subtly rewired.
"""

from __future__ import annotations

import numpy as np
import pytest

from sextant import Index, build
from sextant.index import FORMAT_VERSION


@pytest.fixture
def data() -> np.ndarray:
    return np.random.default_rng(31).normal(size=(300, 20)).astype(np.float32)


@pytest.fixture
def index(data) -> Index:
    return build([f"v{i}" for i in range(len(data))], data,
                 metric="cosine", m=8, ef_construction=80, seed=6)


def test_a_loaded_index_answers_identically(index, data, tmp_path):
    path = tmp_path / "index.npz"
    before = [[(h.id, round(h.distance, 6)) for h in index.search(q, 10, ef=64)]
              for q in data[:30]]

    index.save(path)
    loaded = Index.load(path)

    after = [[(h.id, round(h.distance, 6)) for h in loaded.search(q, 10, ef=64)]
             for q in data[:30]]
    assert after == before


def test_the_graph_itself_comes_back_unchanged(index, tmp_path):
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = Index.load(path)

    assert loaded.graph.max_level == index.graph.max_level
    assert loaded.graph.entry_point == index.graph.entry_point
    assert loaded.graph.levels() == index.graph.levels()
    for level in range(index.graph.max_level + 1):
        for node in range(len(index.graph)):
            assert loaded.graph.neighbours(node, level) == index.graph.neighbours(node, level)


def test_settings_and_ids_survive(index, tmp_path):
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = Index.load(path)

    assert loaded.stats() == index.stats()
    assert set(loaded.ids()) == set(index.ids())
    assert loaded.dim == index.dim
    assert loaded.metric == index.metric


def test_deletions_survive(index, data, tmp_path):
    for i in range(0, 300, 3):
        index.delete(f"v{i}")
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = Index.load(path)

    assert len(loaded) == len(index)
    assert "v0" not in loaded
    assert "v1" in loaded
    for hit in loaded.search(data[0], k=20):
        assert int(hit.id[1:]) % 3 != 0


def test_a_loaded_index_can_still_be_written_to(index, data, tmp_path):
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = Index.load(path)

    fresh = np.zeros(20, dtype=np.float32)
    fresh[0] = 1.0
    loaded.add("added-after-load", fresh)

    assert len(loaded) == 301
    assert loaded.search(fresh, k=1, ef=300)[0].id == "added-after-load"
    # And the vectors that were already there are still findable.
    assert loaded.search(data[5], k=1, ef=300)[0].id == "v5"


def test_an_empty_index_round_trips(tmp_path):
    path = tmp_path / "empty.npz"
    Index(12, seed=1).save(path)
    loaded = Index.load(path)
    assert len(loaded) == 0
    assert loaded.search(np.zeros(12, dtype=np.float32), k=5) == []


def test_saving_creates_missing_directories(index, tmp_path):
    path = tmp_path / "nested" / "deeper" / "index.npz"
    index.save(path)
    assert path.exists()


def test_saving_leaves_no_temporary_file(index, tmp_path):
    index.save(tmp_path / "index.npz")
    assert not list(tmp_path.glob("*.tmp"))


def test_a_future_format_is_refused_rather_than_misread(index, tmp_path, monkeypatch):
    path = tmp_path / "index.npz"
    monkeypatch.setattr("sextant.index.FORMAT_VERSION", FORMAT_VERSION + 1)
    index.save(path)
    monkeypatch.setattr("sextant.index.FORMAT_VERSION", FORMAT_VERSION)

    with pytest.raises(ValueError, match="format version"):
        Index.load(path)


def test_the_adjacency_lists_are_stored_as_arrays_not_json(index, tmp_path):
    """Flat arrays with offsets, so a large graph stays a fraction of the size
    and loads without parsing anything."""
    path = tmp_path / "index.npz"
    index.save(path)
    with np.load(path) as bundle:
        names = set(bundle.files)
    assert "flat_0" in names and "offsets_0" in names and "nodes_0" in names
    assert bundle_dtype(path, "flat_0") == np.int32


def bundle_dtype(path, key):
    with np.load(path) as bundle:
        return bundle[key].dtype
