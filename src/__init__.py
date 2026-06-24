"""
Football Analytics Package
"""

__version__ = "0.1.0"
__author__ = "Alina Letzien"

from .yolo_detector import YOLODetector
from .tracker import ByteTrackTracker
from .team_assigner import TeamAssigner
from .camera_motion import CameraMotionAnalyzer
from .perspective_transformer import PerspectiveTransformer
from .speed_distance_calculator import SpeedDistanceCalculator
from .player_ball_assigner import PlayerBallAssigner

__all__ = [
    "YOLODetector",
    "ByteTrackTracker",
    "TeamAssigner",
    "CameraMotionAnalyzer",
    "PerspectiveTransformer",
    "SpeedDistanceCalculator",
    "PlayerBallAssigner",
]
