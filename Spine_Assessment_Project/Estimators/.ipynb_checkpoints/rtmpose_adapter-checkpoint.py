# estimators/rtmpose_adapter.py
import cv2
import numpy as np
import onnxruntime as ort
from Estimators.base import BasePoseEstimator

class RTMPoseAdapter(BasePoseEstimator):
    
    SIMCC_SPLIT_RATIO = 2.0
    # Attempted: gating keypoints on joint_confidence below a threshold, same
    # rationale as MediaPipeAdapter. REVERTED for the same reason -- without
    # knowing whether this ONNX export's SimCC output is raw logits or
    # softmax-normalized, any fixed threshold is a guess, and in practice it
    # was rejecting genuinely-tracked joints and zeroing real assessments.
    # joint_confidence is still computed below in case this needs revisiting
    # with real calibration data later.

    def __init__(self, model_path="rtmpose-m.onnx"): 
        self.model_path = model_path
        self.session = None 
        # RTMPose outputs standard COCO-17
        self.coco_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  

    def initialize_model(self):
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame: np.ndarray) -> dict:
        if self.session is None:
            self.initialize_model()

        h, w, _ = frame.shape
        # Preprocessing: RTMPose models typically expect 256x192 or 256x256 input
        input_size = (192, 256) 
        resized = cv2.resize(frame, input_size)
        
        img_data = resized.astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        outputs = self.session.run(None, {self.input_name: img_data})
        simcc_x = outputs[0][0]  # shape [17, 384]
        simcc_y = outputs[1][0]  # shape [17, 512]

        x_bins = np.argmax(simcc_x, axis=1)  # shape [17]
        y_bins = np.argmax(simcc_y, axis=1)  # shape [17]
        x_confidence = np.max(simcc_x, axis=1)
        y_confidence = np.max(simcc_y, axis=1)

        keypoints = {}
        success = False

        for idx in self.coco_indices:
         
            joint_confidence = min(x_confidence[idx], y_confidence[idx])

            x_resized = x_bins[idx] / self.SIMCC_SPLIT_RATIO
            y_resized = y_bins[idx] / self.SIMCC_SPLIT_RATIO

            x = (x_resized / input_size[0]) * w
            y = (y_resized / input_size[1]) * h

            keypoints[idx] = [float(x), float(y), 0.0]  # RTMPose is natively 2D
            success = True

        return {"success": success, "keypoints": keypoints}