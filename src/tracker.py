import numpy as np
from typing import Dict, List, Optional
import supervision as sv
from trackers import ByteTrackTracker as _UpstreamByteTrack


class ByteTrackTracker:
    """Track objects across frames using ByteTrack."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        max_ball_interpolation_gap: int = 15,
    ):
        # Migrated from sv.ByteTrack (deprecated since supervision 0.28.0, slated for
        # removal in 0.30.0) to the `trackers` package. Keep this wrapper's own
        # constructor signature/param names stable so config.py and callers don't need
        # to change; translate to the upstream package's names here instead.
        # minimum_matching_threshold -> minimum_iou_threshold is a straight rename,
        # same semantics (IoU threshold for the Hungarian-matching association step).
        #
        # Known behavior change vs. sv.ByteTrack: a brand-new track's tracker_id is
        # always -1 on the frame it's spawned (trackers.core.bytetrack.tracker's
        # _spawn_new_tracks never checks minimum_consecutive_frames at spawn time) —
        # every track's real ID appears starting on its *second* observed frame,
        # regardless of this setting. That startup delay is structural to this
        # library version and can't be configured away. minimum_consecutive_frames
        # only controls how fast a track regains a confirmed ID after being lost and
        # re-matched; pinned to 1 (upstream default 2) for parity with
        # sv.ByteTrack's old default there.
        self.tracker = _UpstreamByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=1,
        )
        self.objects = {}
        self.track_history = {}
        self.max_ball_interpolation_gap = max_ball_interpolation_gap
        self._class_to_id = {"player": 0, "referee": 1, "ball": 2}
        self._id_to_class = {v: k for k, v in self._class_to_id.items()}

    def _normalize_class_name(self, class_name: str) -> str:
        normalized = class_name.lower()
        if normalized in {"person", "player", "goalkeeper"}:
            return "player"
        if normalized in {"ball", "sports ball"}:
            return "ball"
        if normalized == "referee":
            return "referee"
        return normalized

    def update(self, detections: List[Dict]) -> Dict:
        """Update active tracks from detections via ByteTrack."""
        if not detections:
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
            _ = self.tracker.update(empty)
            self.objects = {}
            return self.objects

        xyxy = np.array(
            [[det["x1"], det["y1"], det["x2"], det["y2"]] for det in detections],
            dtype=np.float32,
        )
        confidence = np.array([float(det.get("conf", 0.0)) for det in detections], dtype=np.float32)
        class_id = np.array(
            [
                self._class_to_id.get(
                    self._normalize_class_name(det.get("class_name", "player")),
                    self._class_to_id["player"],
                )
                for det in detections
            ],
            dtype=np.int32,
        )

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self.tracker.update(sv_detections)

        self.objects = {}
        if tracked.confidence is None or tracked.class_id is None or tracked.tracker_id is None:
            return self.objects

        for bbox, conf, cls_id, track_id in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.class_id,
            tracked.tracker_id,
        ):
            # trackers.ByteTrackTracker (unlike sv.ByteTrack) includes detections that
            # haven't yet met minimum_consecutive_frames, marked with tracker_id -1.
            # Drop them rather than let them collide under an `objects[-1]` key.
            if track_id is None or track_id == -1:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            class_name = self._id_to_class.get(int(cls_id), "player")

            self.objects[int(track_id)] = {
                "center": center,
                "bbox": [x1, y1, x2, y2],
                "class_name": class_name,
                "conf": float(conf),
            }
            self.track_history.setdefault(int(track_id), []).append(center)

        return self.objects

    def split_tracks_by_class(self, objects: Optional[Dict] = None) -> Dict[str, Dict]:
        """Split active tracks by semantic class."""
        if objects is None:
            objects = self.objects

        tracks = {
            "players": {},
            "referees": {},
            "ball": {},
        }

        for track_id, track in objects.items():
            class_name = self._normalize_class_name(track.get("class_name", "player"))
            track_payload = {
                "bbox": track["bbox"],
                "center": track["center"],
                "conf": track.get("conf", 0.0),
            }

            if class_name == "ball":
                tracks["ball"][track_id] = track_payload
            elif class_name == "referee":
                tracks["referees"][track_id] = track_payload
            else:
                tracks["players"][track_id] = track_payload

        return tracks

    def interpolate_ball_tracks(self, ball_tracks_per_frame: List[Dict]) -> List[Dict]:
        """Fill short ball detection gaps with linear interpolation."""
        centers = []
        for frame_ball in ball_tracks_per_frame:
            if frame_ball:
                first_id = next(iter(frame_ball))
                centers.append(tuple(frame_ball[first_id]["center"]))
            else:
                centers.append(None)

        known_indices = [i for i, c in enumerate(centers) if c is not None]
        if len(known_indices) < 2:
            return ball_tracks_per_frame

        interpolated = centers[:]
        for left_idx, right_idx in zip(known_indices[:-1], known_indices[1:]):
            left = centers[left_idx]
            right = centers[right_idx]
            gap = right_idx - left_idx
            if gap <= 1:
                continue
            if gap > self.max_ball_interpolation_gap:
                continue  # too long a gap to guess — leave as no ball position

            for step in range(1, gap):
                t = step / float(gap)
                x = int(round(left[0] + (right[0] - left[0]) * t))
                y = int(round(left[1] + (right[1] - left[1]) * t))
                interpolated[left_idx + step] = (x, y)

        output = []
        for frame_idx, frame_ball in enumerate(ball_tracks_per_frame):
            if frame_ball:
                output.append(frame_ball)
                continue
            center = interpolated[frame_idx]
            if center is None:
                output.append({})
                continue
            x, y = center
            output.append({1: {"bbox": [x - 8, y - 8, x + 8, y + 8], "center": center, "conf": 0.0}})
        return output
    