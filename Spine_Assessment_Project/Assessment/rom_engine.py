# assessment/rom_engine.py
import numpy as np

class RomeEngine:
    @staticmethod
    def _calculate_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates the angle in degrees between two 3D vectors."""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        # Dot product formula: cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        # Clip to avoid floating point errors out of range [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        return float(np.degrees(np.arccos(cos_theta)))

    @staticmethod
    def calculate_spine_lateral_bend(keypoints: dict) -> float:
        """
        Measures the angle between the vertical torso vector and a perfect vertical axis.
        """
        # Requires: Shoulders (5,6) and Hips (11,12)
        required = [5, 6, 11, 12]
        if not all(j in keypoints for j in required):
            return 0.0

        mid_shoulder = (np.array(keypoints[5]) + np.array(keypoints[6])) / 2.0
        mid_hip = (np.array(keypoints[11]) + np.array(keypoints[12])) / 2.0

        # Torso vector goes from hips straight up to shoulders
        torso_vector = mid_shoulder - mid_hip
        
        # project on a 2D plane (X, Y) to calculate lateral bend deviation
        vertical_axis = np.array([0.0, -1.0])
        torso_2d = torso_vector[:2]

        angle = RomeEngine._calculate_angle_3d(torso_2d, vertical_axis)
        
        # Determine direction (left vs right bend) based on X deviation
        if torso_vector[0] < 0:
            angle = -angle
            
        return round(angle, 1)

    @staticmethod
    def calculate_knee_flexion(keypoints: dict, side: str = "left") -> float:
        """
        Calculates knee joint flexion angle (interior angle between thigh and calf).
        Left: Hip(11), Knee(13), Ankle(15) | Right: Hip(12), Knee(14), Ankle(16)
        """
        indices = {
            "left": (11, 13, 15),
            "right": (12, 14, 16)
        }
        hip_idx, knee_idx, ankle_idx = indices[side.lower()]

        if not all(j in keypoints for j in [hip_idx, knee_idx, ankle_idx]):
            return 180.0  # Default straight leg angle

        hip = np.array(keypoints[hip_idx])
        knee = np.array(keypoints[knee_idx])
        ankle = np.array(keypoints[ankle_idx])

        thigh_vector = hip - knee
        calf_vector = ankle - knee

        # Calculate internal angle
        angle = RomeEngine._calculate_angle_3d(thigh_vector, calf_vector)
        return round(angle, 1)