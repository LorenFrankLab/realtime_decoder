"""Unit tests for the transition model builders."""

import numpy as np

from realtime_decoder import transitions


def test_sungod_transition_matrix_shape_and_row_sums():
    pos_bins = np.arange(10)
    arm_coords = [[0, 3], [6, 9]]
    bias = 1
    T = transitions.sungod_transition_matrix(pos_bins, arm_coords, bias)

    # square, sized to the number of position bins
    assert T.shape == (len(pos_bins), len(pos_bins))

    # rows that are not entirely zero should sum to 1 (within float
    # tolerance). The gap rows between arms are masked to NaN by
    # apply_no_anim_boundary and then zeroed by the function, so they
    # legitimately sum to 0.
    row_sums = T.sum(axis=1)
    for s in row_sums:
        assert np.isclose(s, 0.0) or np.isclose(s, 1.0)

    # at least the in-arm rows must sum to 1
    in_arm_rows = [r for arm in arm_coords for r in range(arm[0], arm[1] + 1)]
    for r in in_arm_rows:
        assert np.isclose(row_sums[r], 1.0), f"row {r} sums to {row_sums[r]}"


def test_sungod_transition_matrix_gap_rows_are_zero():
    pos_bins = np.arange(10)
    arm_coords = [[0, 3], [6, 9]]
    T = transitions.sungod_transition_matrix(pos_bins, arm_coords, bias=1)
    # bins 4 and 5 are gaps; the corresponding rows and columns should
    # be all zero so transition mass cannot flow through them.
    assert np.all(T[4:6, :] == 0)
    assert np.all(T[:, 4:6] == 0)


def test_sungod_transition_matrix_no_nans():
    pos_bins = np.arange(8)
    arm_coords = [[0, 7]]
    T = transitions.sungod_transition_matrix(pos_bins, arm_coords, bias=1)
    assert not np.any(np.isnan(T))
