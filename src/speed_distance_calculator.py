import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import cv2


class SpeedDistanceCalculator:
    """Calculate player speed and distance covered"""
    
    def __init__(self, fps: float = 30.0, meters_per_pixel: float = 0.1):
        """
        Initialize calculator
        
        Args:
            fps: Frames per second of video
            meters_per_pixel: Conversion factor from pixels to meters
        """
        self.fps = fps
        self.meters_per_pixel = meters_per_pixel
    
    @staticmethod
    def _measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

    @staticmethod
    def _foot_position(bbox: List[float]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def add_speed_and_distance_to_tracks(self, tracks: Dict, frame_window: int = 5):
        """Compute speed/distance for each tracked player over frame windows"""
        players_tracks = tracks.get("players", [])
        total_distance = defaultdict(float)

        number_of_frames = len(players_tracks)
        for frame_num in range(0, number_of_frames, frame_window):
            last_frame = min(frame_num + frame_window, number_of_frames - 1)
            frame_tracks = players_tracks[frame_num]

            for track_id in frame_tracks.keys():
                if track_id not in players_tracks[last_frame]:
                    continue

                start_position = players_tracks[frame_num][track_id].get("position_transformed")
                end_position = players_tracks[last_frame][track_id].get("position_transformed")

                if start_position is None or end_position is None:
                    continue

                distance_covered = self._measure_distance(start_position, end_position)
                time_elapsed = max((last_frame - frame_num) / self.fps, 1e-6)
                speed_mps = distance_covered / time_elapsed
                speed_kmh = speed_mps * 3.6

                total_distance[track_id] += distance_covered

                for frame_batch_idx in range(frame_num, last_frame + 1):
                    if track_id not in players_tracks[frame_batch_idx]:
                        continue
                    players_tracks[frame_batch_idx][track_id]["speed"] = speed_kmh
                    players_tracks[frame_batch_idx][track_id]["distance"] = total_distance[track_id]

    def draw_speed_and_distance(self, frames: List[np.ndarray], tracks: Dict) -> List[np.ndarray]:
        """Draw speed and distance values for players on each frame"""
        output_frames = []
        players_tracks = tracks.get("players", [])

        for frame_num, frame in enumerate(frames):
            output = frame.copy()
            if frame_num < len(players_tracks):
                for _, track_info in players_tracks[frame_num].items():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None or distance is None:
                        continue

                    bbox = track_info["bbox"]
                    x, y = self._foot_position(bbox)
                    y += 35

                    cv2.putText(
                        output,
                        f"{speed:.1f} km/h",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        output,
                        f"{distance:.1f} m",
                        (x, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )

            output_frames.append(output)

        return output_frames
