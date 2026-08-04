def get_pose_estimator(model_name: str):
    """Factory function to instantiate the requested pose estimator."""
    name = model_name.lower()
    
    if "movenet" in name:
        from Estimators.movenet_adapter import MoveNetAdapter
        return MoveNetAdapter()
    elif "rtmpose" in name:
        from Estimators.rtmpose_adapter import RTMPoseAdapter
        return RTMPoseAdapter()
    elif "mediapipe" in name:
        # Check if you have a mediapipe adapter file
        from Estimators.mediapipe_adapter import MediaPipeAdapter
        return MediaPipeAdapter()
    elif "metrabs" in name:
        from Estimators.metrabs_adapter import MeTRAbsAdapter
        return MeTRAbsAdapter()
    else:
        raise ValueError(f"Unknown pose model requested: {model_name}")