# estimators/metrabs_adapter.py
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from Estimators.base import BasePoseEstimator

class MeTRAbsAdapter(BasePoseEstimator):
    """
    This model has no plain 'coco_17' skeleton option. The closest
    match is coco_19 (COCO's 17 core joints plus a synthesized 'neck' and
    'pelv'), but its joint order does not match COCO-17 index numbering. We look up each required joint's real position by name rather than assuming index positions.
    """
    # metrabs_s = small/fast (real-time-viable on CPU), metrabs_l = large/slow/most accurate.
    SKELETON_NAME = "coco_19"

    #   COCO-17 index -> the joint name used inside the coco_19 skeleton
    COCO17_IDX_TO_JOINT_NAME = {
        0: "nose",
        5: "lsho",
        6: "rsho",
        7: "lelb",
        8: "relb",
        9: "lwri",
        10: "rwri",
        11: "lhip",
        12: "rhip",
        13: "lkne",
        14: "rkne",
        15: "lank",
        16: "rank",
    }


    MAX_INFERENCE_DIMENSION = 640

    def __init__(self, variant_url: str = "https://bit.ly/metrabs_s"):
        self.model = None
        self.variant_url = variant_url
        self._joint_index_map = None  # coco17_idx -> position in this model's coco_19 output

    def initialize_model(self):
        
        try:
            self.model = hub.load(self.variant_url)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MeTRAbs model from {self.variant_url}. "
                f"Check network connectivity (TF Hub download required) or try "
                f"again later. Original error: {e}"
            ) from e

       
        joint_names_raw = self.model.per_skeleton_joint_names[self.SKELETON_NAME]
        joint_names = [
            j.numpy().decode("utf-8") if hasattr(j, "numpy") else str(j)
            for j in joint_names_raw
        ]
        name_to_position = {name: i for i, name in enumerate(joint_names)}

        self._joint_index_map = {}
        for coco17_idx, joint_name in self.COCO17_IDX_TO_JOINT_NAME.items():
            if joint_name not in name_to_position:
                raise RuntimeError(
                    f"Expected joint '{joint_name}' not found in '{self.SKELETON_NAME}' "
                    f"skeleton's joint list: {joint_names}"
                )
            self._joint_index_map[coco17_idx] = name_to_position[joint_name]

    def predict(self, frame: np.ndarray) -> dict:
        if self.model is None:
            self.initialize_model()

        orig_h, orig_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       
        scale = 1.0
        longer_edge = max(orig_h, orig_w)
        if longer_edge > self.MAX_INFERENCE_DIMENSION:
            scale = self.MAX_INFERENCE_DIMENSION / longer_edge
            rgb_frame = cv2.resize(rgb_frame, (int(orig_w * scale), int(orig_h * scale)))

        image_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)

        predictions = self.model.detect_poses(image_tensor, skeleton=self.SKELETON_NAME)

        keypoints = {}
        success = False

        # If a person is detected
        if len(predictions['poses2d']) > 0:
            success = True
            
            pose_2d = predictions['poses2d'][0].numpy()  # pixel coords in the (possibly downscaled) frame

            for coco17_idx, position in self._joint_index_map.items():
                x, y = pose_2d[position]
                if scale != 1.0:
                    # Rescale back up to original frame coordinates
                    x = x / scale
                    y = y / scale
                keypoints[coco17_idx] = [float(x), float(y), 0.0]

        return {"success": success, "keypoints": keypoints}
