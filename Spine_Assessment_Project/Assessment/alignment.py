# assessment/alignment.py
import numpy as np

def normalize_and_align_pose(keypoints: dict) -> dict:
    """
    Normalizes a skeleton coordinate dictionary by:
    1. Root-centering on the pelvis (midpoint between hips).
    2. Scaling by the torso length (pelvis to neck/shoulders midpoint) to remove distance bias.
    """
    # 17-point COCO Indices:
    # 11: Left Hip, 12: Right Hip, 5: Left Shoulder, 6: Right Shoulder
    required_joints = [5, 6, 11, 12]
    if not all(j in keypoints for j in required_joints):
        return keypoints  # Can't normalize if core hips/shoulders are missing

    # Extract coordinates as numpy arrays
    l_shoulder = np.array(keypoints[5])
    r_shoulder = np.array(keypoints[6])
    l_hip = np.array(keypoints[11])
    r_hip = np.array(keypoints[12])

    # 1. Compute root center (Pelvis midpoint)
    pelvis = (l_hip + r_hip) / 2.0

    # Translate all joints so that the Pelvis is the origin (0, 0, 0)
    root_centered_keypoints = {}
    for joint_idx, coords in keypoints.items():
        root_centered_keypoints[joint_idx] = np.array(coords) - pelvis

    # 2. Scale by Torso Length to handle scale differences
    # Calculate shoulder midpoint (neck base)
    neck = (l_shoulder + r_shoulder) / 2.0
    # Torso length is the distance from pelvis to neck
    torso_length = np.linalg.norm(neck - pelvis)

    if torso_length == 0:
        torso_length = 1.0

    # Divide all coordinates by torso length to achieve scale invariance
    normalized_keypoints = {}
    for joint_idx, coords in root_centered_keypoints.items():
        # Scale X, Y, and Z (if Z is present)
        normalized_keypoints[joint_idx] = (coords / torso_length).tolist()

    return normalized_keypoints


class TemporalSmoother:
    """
    smoothing out high-frequency jitter 
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = {}

    def smooth(self, keypoints: dict) -> dict:
        if not keypoints:
            return keypoints

        smoothed_keypoints = {}
        for joint_idx, coords in keypoints.items():
            if joint_idx not in self.history:
                self.history[joint_idx] = []
            
            # Append current coordinate
            self.history[joint_idx].append(coords)
            
            # Keep history within sliding window size
            if len(self.history[joint_idx]) > self.window_size:
                self.history[joint_idx].pop(0)
            
            # Calculate mean position over history window
            coords_matrix = np.array(self.history[joint_idx])
            smoothed_coords = np.mean(coords_matrix, axis=0)
            smoothed_keypoints[joint_idx] = smoothed_coords.tolist()

        return smoothed_keypoints