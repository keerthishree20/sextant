"""The metrics. Everything above them inherits whatever they get wrong."""

from __future__ import annotations

import numpy as np
import pytest

from sextant import distance


def test_l2_matches_a_direct_computation():
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    candidates = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], dtype=np.float32)
    got = distance.l2_squared(query, candidates)
    want = np.array([0.0, 27.0, 14.0], dtype=np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_inner_product_is_negated_so_smaller_is_nearer():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    got = distance.negative_inner_product(query, candidates)
    assert got[0] < got[1] < got[2]


def test_normalise_makes_unit_rows():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-6.0, 8.0]], dtype=np.float32)
    unit = distance.normalise(vectors)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, rtol=1e-6)


def test_a_zero_vector_normalises_to_zero_not_to_nan():
    """A zero row has no direction. Dividing by its length would poison every
    later comparison with NaN."""
    unit = distance.normalise(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert np.all(np.isfinite(unit))
    assert np.all(unit == 0)


def test_cosine_reports_a_distance_from_zero_to_two():
    identical = distance.report("cosine", distance.negative_inner_product(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32))[0])
    opposite = distance.report("cosine", distance.negative_inner_product(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[-1.0, 0.0]], dtype=np.float32))[0])
    orthogonal = distance.report("cosine", distance.negative_inner_product(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32))[0])

    assert identical == pytest.approx(0.0, abs=1e-6)
    assert orthogonal == pytest.approx(1.0, abs=1e-6)
    assert opposite == pytest.approx(2.0, abs=1e-6)


def test_l2_reports_the_real_distance_not_the_squared_one():
    assert distance.report("l2", 9.0) == pytest.approx(3.0)
    # Floating point can push a squared distance a hair below zero.
    assert distance.report("l2", -1e-9) == 0.0


def test_only_cosine_normalises():
    assert distance.normalises("cosine") is True
    assert distance.normalises("l2") is False
    assert distance.normalises("inner_product") is False


def test_an_unknown_metric_names_the_real_ones():
    with pytest.raises(ValueError, match="cosine, inner_product, l2"):
        distance.function_for("manhattan")
