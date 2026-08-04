"""
Stage 2 tests: get_pose_estimator() factory routing.

"""
import pytest

from Estimators import get_pose_estimator


def test_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError):
        get_pose_estimator("NotARealModel")


@pytest.mark.parametrize("name", ["MoveNet", "movenet", "MOVENET"])
def test_movenet_name_variants_resolve(name):
    tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")
    pytest.importorskip("tensorflow_hub", reason="tensorflow_hub not installed")
    from Estimators.movenet_adapter import MoveNetAdapter
    estimator = get_pose_estimator(name)
    assert isinstance(estimator, MoveNetAdapter)
    assert estimator.module is None


def test_rtmpose_resolves_and_does_not_import_tensorflow():
    """Regression test for the bug where rtmpose_adapter.py accidentally
    imported MoveNetAdapter (and therefore TensorFlow) at module load time.
    """
    pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
    import sys
    tf_was_loaded_before = "tensorflow" in sys.modules

    from Estimators.rtmpose_adapter import RTMPoseAdapter
    estimator = get_pose_estimator("RTMPose")
    assert isinstance(estimator, RTMPoseAdapter)

    tf_was_loaded_after = "tensorflow" in sys.modules
    if not tf_was_loaded_before:
        assert not tf_was_loaded_after, (
            "Selecting RTMPose should not pull TensorFlow into sys.modules"
        )


def test_mediapipe_resolves():
    pytest.importorskip("mediapipe", reason="mediapipe not installed")
    from Estimators.mediapipe_adapter import MediaPipeAdapter
    estimator = get_pose_estimator("MediaPipe")
    assert isinstance(estimator, MediaPipeAdapter)


def test_metrabs_resolves():
    """Regression test for the bug where MeTRAbs wasn't registered in the
    factory at all and selecting it in the UI raised ValueError."""
    pytest.importorskip("tensorflow", reason="tensorflow not installed")
    pytest.importorskip("tensorflow_hub", reason="tensorflow_hub not installed")
    from Estimators.metrabs_adapter import MeTRAbsAdapter
    estimator = get_pose_estimator("MeTRAbs")
    assert isinstance(estimator, MeTRAbsAdapter)


def test_all_adapters_implement_required_interface():
    """Every adapter must expose initialize_model() and predict(),
    per the BasePoseEstimator contract."""
    from Estimators.base import BasePoseEstimator
    import inspect

    adapter_modules = []
    try:
        from Estimators.movenet_adapter import MoveNetAdapter
        adapter_modules.append(MoveNetAdapter)
    except ImportError:
        pass
    try:
        from Estimators.rtmpose_adapter import RTMPoseAdapter
        adapter_modules.append(RTMPoseAdapter)
    except ImportError:
        pass
    try:
        from Estimators.mediapipe_adapter import MediaPipeAdapter
        adapter_modules.append(MediaPipeAdapter)
    except ImportError:
        pass
    try:
        from Estimators.metrabs_adapter import MeTRAbsAdapter
        adapter_modules.append(MeTRAbsAdapter)
    except ImportError:
        pass

    assert adapter_modules, "No adapters could be imported -- check dependencies"

    for cls in adapter_modules:
        assert issubclass(cls, BasePoseEstimator)
        assert not inspect.isabstract(cls), f"{cls.__name__} is missing a required method"
