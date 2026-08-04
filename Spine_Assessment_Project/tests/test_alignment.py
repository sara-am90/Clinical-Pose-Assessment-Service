"""
Stage 1 tests: normalize_and_align_pose() and TemporalSmoother.

"""
import numpy as np
import pytest

from Assessment.alignment import normalize_and_align_pose, TemporalSmoother


def make_upright_pose():
   
    return {
        0:  [250.0, 50.0, 0.0],   # nose
        5:  [200.0, 100.0, 0.0],  # left shoulder
        6:  [300.0, 100.0, 0.0],  # right shoulder
        11: [220.0, 300.0, 0.0],  # left hip
        12: [280.0, 300.0, 0.0],  # right hip
        13: [220.0, 400.0, 0.0],  # left knee
        14: [280.0, 400.0, 0.0],  # right knee
        15: [220.0, 500.0, 0.0],  # left ankle
        16: [280.0, 500.0, 0.0],  # right ankle
    }


def test_normalize_centers_pelvis_at_origin():
    pose = make_upright_pose()
    result = normalize_and_align_pose(pose)
    pelvis_after = (np.array(result[11]) + np.array(result[12])) / 2.0
    assert np.allclose(pelvis_after, [0.0, 0.0, 0.0], atol=1e-6)


def test_normalize_is_scale_invariant():
    pose = make_upright_pose()
    scaled_pose = {k: [c * 2 for c in v] for k, v in pose.items()}

    result_a = normalize_and_align_pose(pose)
    result_b = normalize_and_align_pose(scaled_pose)

    for joint in pose:
        assert np.allclose(result_a[joint], result_b[joint], atol=1e-6)


def test_normalize_missing_core_joints_returns_input_unchanged():
    """If shoulders/hips aren't detected, function should pass keypoints through."""
    pose = {0: [250.0, 50.0, 0.0]}  # only nose, no shoulders/hips
    result = normalize_and_align_pose(pose)
    assert result == pose


def test_normalize_handles_zero_torso_length_without_crashing():
    pose = {
        5: [250.0, 200.0, 0.0],
        6: [250.0, 200.0, 0.0],
        11: [250.0, 200.0, 0.0],
        12: [250.0, 200.0, 0.0],
    }
    result = normalize_and_align_pose(pose)  
    assert all(np.isfinite(coords).all() for coords in result.values())


def test_smoother_averages_over_window():
    smoother = TemporalSmoother(window_size=3)
    smoother.smooth({0: [0.0, 0.0, 0.0]})
    smoother.smooth({0: [10.0, 0.0, 0.0]})
    result = smoother.smooth({0: [20.0, 0.0, 0.0]})
    # average of 0, 10, 20 = 10
    assert np.allclose(result[0], [10.0, 0.0, 0.0])


def test_smoother_window_slides_correctly():
    smoother = TemporalSmoother(window_size=2)
    smoother.smooth({0: [0.0, 0.0, 0.0]})
    smoother.smooth({0: [10.0, 0.0, 0.0]})
    result = smoother.smooth({0: [20.0, 0.0, 0.0]})
    # window_size=2, oldest (0.0) should have been dropped -> avg(10, 20) = 15
    assert np.allclose(result[0], [15.0, 0.0, 0.0])


def test_smoother_empty_input_returns_empty():
    smoother = TemporalSmoother()
    assert smoother.smooth({}) == {}
