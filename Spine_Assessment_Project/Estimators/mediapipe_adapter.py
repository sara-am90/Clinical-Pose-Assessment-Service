# estimators/mediapipe_adapter.py
import cv2
import mediapipe as mp
import numpy as np
from Estimators.base import BasePoseEstimator

class MediaPipeAdapter(BasePoseEstimator):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = None
        # COCO-17 mapping indices for MediaPipe
        self.mp_to_coco = {
            0: 0,    # nose -> nose
            5: 11,   # left_shoulder -> left_shoulder
            6: 12,   # right_shoulder -> right_shoulder
            11: 23,  # left_hip -> left_hip
            12: 24,  # right_hip -> right_hip
            13: 25,  # left_knee -> left_knee
            14: 26,  # right_knee -> right_knee
            15: 27,  # left_ankle -> left_ankle
            16: 28   # right_ankle -> right_ankle
        }

    def initialize_model(self):
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,  # Smooths out frame-to-frame jitter
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def predict(self, frame: np.ndarray) -> dict:
        if self.pose is None:
            self.initialize_model()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        keypoints = {}
        success = False

        if results.pose_landmarks:
            success = True
            landmarks = results.pose_landmarks.landmark
            
            # Map MediaPipe relative coordinates to pixel coordinates for consistency
            for coco_idx, mp_idx in self.mp_to_coco.items():
                lm = landmarks[mp_idx]
                # Convert normalized [0, 1] to pixel coordinates
                pixel_x = lm.x * w
                pixel_y = lm.y * h
                # MediaPipe z represents relative depth
                pixel_z = lm.z * w  
                
                keypoints[coco_idx] = [pixel_x, pixel_y, pixel_z]

        return {"success": success, "keypoints": keypoints}