"""Unit tests for realtime_decoder.position."""

import numpy as np
import pytest

from realtime_decoder import position, datatypes


# ---------------------------------------------------------------------------
# PositionBinStruct
# ---------------------------------------------------------------------------


def test_position_bin_struct_edges_and_centers():
    s = position.PositionBinStruct(0, 10, 5)
    np.testing.assert_allclose(s.pos_bin_edges, [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(s.pos_bin_centers, [1, 3, 5, 7, 9])
    assert s.pos_bin_delta == 2.0
    assert s.num_bins == 5


@pytest.mark.parametrize("pos,expected_bin", [
    (0.0, 0),
    (1.99, 0),
    (2.0, 1),
    (5.5, 2),
    (9.99, 4),
])
def test_get_bin_within_range(pos, expected_bin):
    s = position.PositionBinStruct(0, 10, 5)
    assert s.get_bin(pos) == expected_bin


# ---------------------------------------------------------------------------
# TrodesPositionMapper
# ---------------------------------------------------------------------------


def _camera_point(*, segment, position_on_segment):
    """Build a CameraModulePoint with the bare minimum the mapper reads."""
    return datatypes.CameraModulePoint(
        timestamp=0,
        segment=segment,
        position=position_on_segment,
        x=0.0, y=0.0, x2=0.0, y2=0.0,
        t_recv_data=0,
    )


def test_position_mapper_basic():
    # 2 arms: segment 0 -> arm 0 (bins 0..3), segment 1 -> arm 1 (bins 5..8)
    mapper = position.TrodesPositionMapper(
        arm_ids=[0, 1],
        arm_coords=[[0, 3], [5, 8]],
    )
    # arm 0 has 4 bins; normalized edges [0, .25, .5, .75, 1]
    assert mapper.map_position(_camera_point(segment=0, position_on_segment=0.0)) == 0
    assert mapper.map_position(_camera_point(segment=0, position_on_segment=0.5)) == 2
    # exact upper edge clamps to the last bin per the inclusive-upper rule
    assert mapper.map_position(_camera_point(segment=0, position_on_segment=1.0)) == 3
    # arm 1 starts at bin 5
    assert mapper.map_position(_camera_point(segment=1, position_on_segment=0.0)) == 5
    assert mapper.map_position(_camera_point(segment=1, position_on_segment=1.0)) == 8


def test_position_mapper_above_one_clamps():
    # numerical noise that pushes segment position slightly above 1.0
    # should still land in the last bin rather than crashing.
    mapper = position.TrodesPositionMapper(
        arm_ids=[0],
        arm_coords=[[0, 4]],
    )
    assert mapper.map_position(_camera_point(segment=0, position_on_segment=1.0001)) == 4


# ---------------------------------------------------------------------------
# KinematicsEstimator
# ---------------------------------------------------------------------------


def test_kinematics_first_sample_returns_zero_speed():
    est = position.KinematicsEstimator(
        scale_factor=1.0, dt=1.0,
        xfilter=[1.0], yfilter=[1.0], speedfilter=[1.0],
    )
    x, y, s = est.compute_kinematics(10.0, 20.0)
    assert (x, y, s) == (10.0, 20.0, 0)


def test_kinematics_speed_unsmoothed_matches_euclid_distance():
    est = position.KinematicsEstimator(
        scale_factor=1.0, dt=0.5,
        xfilter=[1.0], yfilter=[1.0], speedfilter=[1.0],
    )
    est.compute_kinematics(0.0, 0.0)  # prime
    x, y, s = est.compute_kinematics(3.0, 4.0)
    # 5 units over 0.5s -> 10 units/sec
    assert (x, y) == (3.0, 4.0)
    assert np.isclose(s, 10.0)


def test_kinematics_smoothing_applies_fir():
    # 3-tap moving average: smoothed value of the third sample should
    # equal the average of the last three inputs (* scale).
    est = position.KinematicsEstimator(
        scale_factor=1.0, dt=1.0,
        xfilter=[1/3, 1/3, 1/3],
        yfilter=[1/3, 1/3, 1/3],
        speedfilter=[1.0],
    )
    est.compute_kinematics(0.0, 0.0)  # prime, returned raw
    est.compute_kinematics(6.0, 0.0, smooth_x=True)  # buf=[6,0,0]
    x, _, _ = est.compute_kinematics(9.0, 0.0, smooth_x=True)  # buf=[9,6,0]
    assert np.isclose(x, 5.0)  # (9+6+0)/3
