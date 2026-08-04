# estimators/base.py
from abc import ABC, abstractmethod
import numpy as np

class BasePoseEstimator(ABC):
    """
    Abstract Base Class:it enforces a common interface for all pose estimation models.
    """
    
    @abstractmethod
    def initialize_model(self):
        """Load weights and initializes the model."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> dict:
        """
        Processes a single BGR frame and returns a standardized dictionary.
        
        The output format that we expect too have:
        {
            "success": bool,
            "keypoints": {
                0: [x, y, z_opt],   # nose (COCO-17 index)
                5: [x, y, z_opt],   # left_shoulder
                6: [x, y, z_opt],   # right_shoulder
                11: [x, y, z_opt],  # left_hip
                12: [x, y, z_opt],  # right_hip
                13: [x, y, z_opt],  # left_knee
                14: [x, y, z_opt],  # right_knee
                15: [x, y, z_opt],  # left_ankle
                16: [x, y, z_opt],  # right_ankle
                ...
            }
        }

        """
        pass