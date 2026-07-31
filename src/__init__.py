"""
Football Analytics Package
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "Alina Letzien"

__all__ = [
    "YOLODetector",
    "ByteTrackTracker",
    "TeamAssigner",
    "CameraMotionAnalyzer",
    "PerspectiveTransformer",
    "SpeedDistanceCalculator",
    "PlayerBallAssigner",
]

# name -> submodule holding it. Every codepath in this repo already imports
# submodules directly (e.g. `from src.team_assigner import TeamAssigner`), so
# these only matter for `from src import X`-style callers. Lazily resolving
# them here (PEP 562) keeps `import src.team_assigner` from also pulling in
# yolo_detector's `ultralytics`/tracker's `supervision` deps, which broke CI's
# fast-tests job (it only installs numpy/opencv/scikit-learn).
_SUBMODULE_BY_NAME = {
    "YOLODetector": "yolo_detector",
    "ByteTrackTracker": "tracker",
    "TeamAssigner": "team_assigner",
    "CameraMotionAnalyzer": "camera_motion",
    "PerspectiveTransformer": "perspective_transformer",
    "SpeedDistanceCalculator": "speed_distance_calculator",
    "PlayerBallAssigner": "player_ball_assigner",
}


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{submodule}", __name__), name)
