"""
Stage 1 tests: RomeEngine angle calculations.


"""
import numpy as np
import pytest

from Assessment.rom_engine import RomeEngine


def test_angle_between_orthogonal_vectors_is_90():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert RomeEngine._calculate_angle_3d(v1, v2) == pytest.approx(90.0)


def test_angle_between_identical_vectors_is_0():
    v1 = np.array([1.0, 2.0, 3.0])
    assert RomeEngine._calculate_angle_3d(v1, v1) == pytest.approx(0.0, abs=1e-6)


def test_angle_between_opposite_vectors_is_180():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert RomeEngine._calculate_angle_3d(v1, v2) == pytest.approx(180.0)


def test_angle_handles_zero_length_vector_without_crashing():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert RomeEngine._calculate_angle_3d(v1, v2) == 0.0


# ---------- calculate_spine_lateral_bend ----------

def test_spine_bend_zero_for_perfectly_upright_torso():
    # shoulders directly above hips, centered on same x
    pose = {
        5: [200.0, 100.0, 0.0], 6: [300.0, 100.0, 0.0],   # shoulders, midpoint x=250
        11: [220.0, 300.0, 0.0], 12: [280.0, 300.0, 0.0],  # hips, midpoint x=250
    }
    angle = RomeEngine.calculate_spine_lateral_bend(pose)
    assert angle == pytest.approx(0.0, abs=0.5)


def test_spine_bend_sign_flips_for_left_vs_right_lean():
    # shoulders shifted left of hip midpoint -> one sign
    left_lean = {
        5: [100.0, 100.0, 0.0], 6: [200.0, 100.0, 0.0],   # midpoint x=150
        11: [220.0, 300.0, 0.0], 12: [280.0, 300.0, 0.0],  # midpoint x=250
    }
    # shoulders shifted right of hip midpoint -> opposite sign
    right_lean = {
        5: [300.0, 100.0, 0.0], 6: [400.0, 100.0, 0.0],   # midpoint x=350
        11: [220.0, 300.0, 0.0], 12: [280.0, 300.0, 0.0],  # midpoint x=250
    }
    left_angle = RomeEngine.calculate_spine_lateral_bend(left_lean)
    right_angle = RomeEngine.calculate_spine_lateral_bend(right_lean)
    assert left_angle < 0 < right_angle
    assert abs(left_angle) == pytest.approx(abs(right_angle), abs=0.5)


def test_spine_bend_missing_joints_returns_zero():
    assert RomeEngine.calculate_spine_lateral_bend({5: [0, 0, 0]}) == 0.0


# ---------- calculate_knee_flexion ----------

def test_knee_flexion_straight_leg_is_180():
    # hip, knee, ankle all in a straight vertical line
    pose = {
        11: [250.0, 300.0, 0.0],  # left hip
        13: [250.0, 400.0, 0.0],  # left knee
        15: [250.0, 500.0, 0.0],  # left ankle
    }
    angle = RomeEngine.calculate_knee_flexion(pose, side="left")
    assert angle == pytest.approx(180.0, abs=0.5)


def test_knee_flexion_90_degree_bend():
    # thigh straight up from knee, calf straight forward from knee -> 90 deg interior angle
    pose = {
        11: [250.0, 300.0, 0.0],  # hip directly above knee
        13: [250.0, 400.0, 0.0],  # knee
        15: [350.0, 400.0, 0.0],  # ankle directly to the side of knee
    }
    angle = RomeEngine.calculate_knee_flexion(pose, side="left")
    assert angle == pytest.approx(90.0, abs=0.5)


def test_knee_flexion_right_side_uses_correct_joint_indices():
    pose = {
        12: [250.0, 300.0, 0.0],  # right hip
        14: [250.0, 400.0, 0.0],  # right knee
        16: [250.0, 500.0, 0.0],  # right ankle
    }
    angle = RomeEngine.calculate_knee_flexion(pose, side="right")
    assert angle == pytest.approx(180.0, abs=0.5)


def test_knee_flexion_missing_joints_defaults_to_180():
    """Documented fallback: treat undetected leg as straight rather than 0."""
    assert RomeEngine.calculate_knee_flexion({}, side="left") == 180.0
