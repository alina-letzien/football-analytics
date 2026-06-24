import cv2
import numpy as np
from typing import List, Optional, Tuple


class PerspectiveTransformer:
    """Transform 2D image coordinates to field coordinates"""

    def __init__(self, field_width: float = 120.0, field_height: float = 80.0):
        self.field_width = field_width
        self.field_height = field_height
        self.transformation_matrix = None
        self.inverse_matrix = None

    def set_calibration_points(
        self,
        src_points: List[Tuple[int, int]],
        dst_points: List[Tuple[float, float]],
    ) -> None:
        """
        Set calibration points for perspective transformation.

        Args:
            src_points: 4 points in image coordinates (pixels)
            dst_points: 4 corresponding points in field coordinates (meters)
        """
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self.transformation_matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    def pixel_to_field(self, point: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to field coordinates (meters). Returns None if uncalibrated."""
        if self.transformation_matrix is None:
            return None

        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.transformation_matrix)
        x, y = transformed[0][0]
        return float(x), float(y)

    def field_to_pixel(self, point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Convert field coordinates (meters) to pixel coordinates. Returns None if uncalibrated."""
        if self.inverse_matrix is None:
            return None

        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.inverse_matrix)
        x, y = transformed[0][0]
        return int(x), int(y)

    def add_transformed_position_to_tracks(self, tracks: dict) -> None:
        """Add perspective-transformed positions to track dictionaries."""
        for _, object_tracks in tracks.items():
            for _, frame_tracks in enumerate(object_tracks):
                for _, track_info in frame_tracks.items():
                    position = track_info.get("position_adjusted")
                    if position is None:
                        track_info["position_transformed"] = None
                        continue

                    transformed = self.pixel_to_field((int(position[0]), int(position[1])))
                    if transformed is None:
                        track_info["position_transformed"] = None
                        continue

                    track_info["position_transformed"] = (float(transformed[0]), float(transformed[1]))
