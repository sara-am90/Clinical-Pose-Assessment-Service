# Assessment/registry.py
from Assessment.pose_detector import (
    ShoulderPainfulArcAssessment,
    SpineAssessment,
    HipAssessment,
    AnkleAssessment,
    KneeAssessment
)

# Map strings to Class references
ASSESSMENT_REGISTRY = {
    "Shoulder Abduction (Painful Arc)": ShoulderPainfulArcAssessment,
    "Spine Lateral Flexion": SpineAssessment,
    "Hip Abduction & Adduction": HipAssessment,
    "Ankle Range of Motion": AnkleAssessment,
    "Knee Flexion": KneeAssessment
}

def get_assessment(name: str):
    """Factory function: instantiates and returns the selected class."""
    if name not in ASSESSMENT_REGISTRY:
        raise ValueError(f"Unknown assessment: {name}")
    
    return ASSESSMENT_REGISTRY[name]()