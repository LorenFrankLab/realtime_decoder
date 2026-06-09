"""Unit tests for realtime_decoder.utils helpers."""

import numpy as np
import pytest

from realtime_decoder import utils


def test_normalize_to_probability_simple():
    dist = np.array([1.0, 2.0, 3.0, 4.0])
    out = utils.normalize_to_probability(dist)
    assert np.isclose(out.sum(), 1.0)
    np.testing.assert_allclose(out, dist / dist.sum())


def test_normalize_to_probability_ignores_nan():
    # np.nansum is used internally, so NaN entries should not bias
    # the normalization of the other entries.
    dist = np.array([1.0, np.nan, 3.0])
    out = utils.normalize_to_probability(dist)
    # the two finite entries should sum to 1 between them (NaN propagates
    # to its own bin but the divisor was sum(1+3)=4)
    finite = out[np.isfinite(out)]
    assert np.isclose(finite.sum(), 1.0)


def test_estimate_new_stats_matches_numpy():
    # Welford's online stats should converge to numpy's batch stats.
    rng = np.random.default_rng(0)
    values = rng.standard_normal(500)
    mean = 0.0
    M2 = 0.0
    count = 0
    for v in values:
        mean, M2, count = utils.estimate_new_stats(v, mean, M2, count)
    # variance from M2; compare to numpy population variance (ddof=0)
    var = M2 / count
    assert np.isclose(mean, values.mean(), atol=1e-12)
    assert np.isclose(var, values.var(), atol=1e-12)


def test_apply_no_anim_boundary_2d_fills_gaps():
    # arm_coords [[0,3],[6,9]] => bins 4 and 5 are "no animal" gaps.
    x_bins = np.arange(10)
    arm_coords = [[0, 3], [6, 9]]
    image = np.ones((10, 10))
    out = utils.apply_no_anim_boundary(x_bins, arm_coords, image, fill=0)
    # gap rows and columns should be zeroed
    assert np.all(out[4:6, :] == 0)
    assert np.all(out[:, 4:6] == 0)
    # non-gap interior should still be 1
    assert out[1, 1] == 1
    assert out[7, 7] == 1


def test_apply_no_anim_boundary_1d_fills_gaps():
    x_bins = np.arange(10)
    arm_coords = [[0, 3], [6, 9]]
    image = np.ones(10)
    out = utils.apply_no_anim_boundary(x_bins, arm_coords, image, fill=-1)
    assert np.all(out[4:6] == -1)
    assert out[0] == 1 and out[9] == 1
