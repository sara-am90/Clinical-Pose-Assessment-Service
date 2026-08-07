# estimators/movenet_adapter.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from Estimators.base import BasePoseEstimator

class MoveNetAdapter(BasePoseEstimator):
    def __init__(self):
        self.module = None
        # MoveNet natively outputs COCO-17 points, so mapping is 1:1
        self.coco_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # Key clinical nodes

    def initialize_model(self):

        self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        self.model = self.module.signatures['serving_default']

    def predict(self, frame: np.ndarray) -> dict:
        if self.module is None:
            self.initialize_model()

        h, w, _ = frame.shape
        # MoveNet Thunder expects a 256x256 image.
        input_image = tf.image.resize_with_pad(tf.convert_to_tensor(frame), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)

        outputs = self.model(input_image)
        # Output shape is [1, 1, 17, 3] -> [y, x, confidence]
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]

        keypoints = {}
        success = False

        if len(keypoints_with_scores) > 0:
            success = True
            for idx in self.coco_indices:
                y, x, confidence = keypoints_with_scores[idx]
                if confidence > 0.3:  # Confidence threshold filter
                    pixel_x = x * w
                    pixel_y = y * h
                    # MoveNet is purely 2D, set Z to 0
                    keypoints[idx] = [pixel_x, pixel_y, 0.0]

        return {"success": success, "keypoints": keypoints}