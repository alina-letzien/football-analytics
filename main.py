import cv2
import numpy as np
import os
import pickle

from config import YOLO_CONFIG, TRACKER_CONFIG, TEAM_CONFIG, FIELD_CONFIG, SPEED_CONFIG, VIDEO_CONFIG, POSSESSION_CONFIG, COLORS
from src.yolo_detector import YOLODetector
from src.tracker import ByteTrackTracker
from src.team_assigner import TeamAssigner
from src.camera_motion import CameraMotionAnalyzer
from src.perspective_transformer import PerspectiveTransformer
from src.speed_distance_calculator import SpeedDistanceCalculator
from src.player_ball_assigner import PlayerBallAssigner


class FootballAnalyzer:
    """Main football analysis pipeline"""
    
    def __init__(
        self,
        video_path: str,
        output_path: str = VIDEO_CONFIG["output_path"],
        model_path: str = YOLO_CONFIG["model_path"],
        device: str = YOLO_CONFIG["device"],
        read_from_stub: bool = VIDEO_CONFIG["read_from_stub"],
        stub_dir: str = VIDEO_CONFIG["stub_dir"],
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.read_from_stub = read_from_stub
        self.stub_dir = stub_dir

        os.makedirs(self.stub_dir, exist_ok=True)

        print("Initializing YOLO detector...")
        self.detector = YOLODetector(
            model_path=model_path,
            device=device,
            conf=YOLO_CONFIG["confidence_threshold"],
            iou=YOLO_CONFIG["iou_threshold"],
        )

        print("Initializing tracker...")
        self.tracker = ByteTrackTracker(
            track_activation_threshold=TRACKER_CONFIG["track_activation_threshold"],
            lost_track_buffer=TRACKER_CONFIG["lost_track_buffer"],
            minimum_matching_threshold=TRACKER_CONFIG["minimum_matching_threshold"],
            frame_rate=TRACKER_CONFIG["frame_rate"],
        )

        print("Initializing team assigner...")
        self.team_assigner = TeamAssigner(
            n_teams=TEAM_CONFIG["n_teams"],
            random_state=TEAM_CONFIG["kmeans_random_state"],
            color_sample_frames=TEAM_CONFIG["color_sample_frames"],
            min_confidence_threshold=TEAM_CONFIG["min_confidence_threshold"],
            team_color_overrides=TEAM_CONFIG["team_color_overrides"],
        )

        print("Initializing player-ball assigner...")
        self.player_ball_assigner = PlayerBallAssigner(
            max_player_ball_distance=POSSESSION_CONFIG["max_player_ball_distance"]
        )

        print("Initializing camera motion analyzer...")
        self.camera_motion = CameraMotionAnalyzer()

        print("Initializing perspective transformer...")
        self.perspective = PerspectiveTransformer(
            field_width=FIELD_CONFIG["field_width"],
            field_height=FIELD_CONFIG["field_height"],
        )

        print("Initializing speed/distance calculator...")
        self.speed_calculator = SpeedDistanceCalculator(
            fps=SPEED_CONFIG["fps"],
            meters_per_pixel=SPEED_CONFIG["meters_per_pixel"],
        )

    @staticmethod
    def _read_video(video_path: str):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        fps = cap.get(cv2.CAP_PROP_FPS) or SPEED_CONFIG["fps"]
        cap.release()
        return frames, fps

    @staticmethod
    def _save_video(frames, output_path: str, fps: float, codec: str = VIDEO_CONFIG["output_codec"]):
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec) # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()

    @staticmethod
    def _foot_position(bbox):
        x1, _, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    @staticmethod
    def _center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _tracks_stub_path(self):
        return os.path.join(self.stub_dir, "track_stubs.pkl")

    def _camera_stub_path(self):
        return os.path.join(self.stub_dir, "camera_movement_stubs.pkl")

    def _build_tracks(self, video_frames):
        stub_path = self._tracks_stub_path()
        if self.read_from_stub and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, frame in enumerate(video_frames):
            detections = self.detector.detect(frame)
            tracker_input = detections["players"] + detections["referees"] + detections["balls"]
            active_tracks = self.tracker.update(tracker_input)
            split_tracks = self.tracker.split_tracks_by_class(active_tracks)

            tracks["players"].append(split_tracks["players"])
            tracks["referees"].append(split_tracks["referees"])
            tracks["ball"].append(split_tracks["ball"])

            if frame_num % 30 == 0:
                print(f"Tracked {frame_num}/{len(video_frames)} frames")

        with open(stub_path, "wb") as f:
            pickle.dump(tracks, f)

        return tracks

    def _add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    bbox = track_info["bbox"]
                    if object_name == "ball":
                        position = self._center_of_bbox(bbox)
                    else:
                        position = self._foot_position(bbox)
                    tracks[object_name][frame_num][track_id]["position"] = position

    def _add_fallback_transformed_positions(self, tracks):
        # If no perspective calibration is set, convert adjusted pixels to meters.
        scale = self.speed_calculator.meters_per_pixel
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    adjusted = track_info.get("position_adjusted")
                    if adjusted is None:
                        tracks[object_name][frame_num][track_id]["position_transformed"] = None
                        continue
                    tracks[object_name][frame_num][track_id]["position_transformed"] = (
                        adjusted[0] * scale,
                        adjusted[1] * scale,
                    )

    def _draw_tracks(self, frame, frame_tracks):
        output = frame.copy()

        for track_id, track_info in frame_tracks.get("players", {}).items():
            x1, y1, x2, y2 = map(int, track_info["bbox"])
            color = track_info.get("team_color", COLORS["player"])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"P{track_id} T{track_info.get('team', 0)}"
            if track_info.get("has_ball"):
                label += " *"
            cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for track_id, track_info in frame_tracks.get("referees", {}).items():
            x1, y1, x2, y2 = map(int, track_info["bbox"])
            cv2.rectangle(output, (x1, y1), (x2, y2), COLORS["referee"], 2)
            cv2.putText(output, f"R{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["referee"], 2)

        for _, track_info in frame_tracks.get("ball", {}).items():
            x1, y1, x2, y2 = map(int, track_info["bbox"])
            cv2.rectangle(output, (x1, y1), (x2, y2), COLORS["ball"], 2)
            cv2.putText(output, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["ball"], 2)

        return output

    @staticmethod
    def _draw_team_ball_control(frame, frame_num, team_ball_control):
        output = frame.copy()
        if len(team_ball_control) == 0:
            return output

        history = np.array(team_ball_control[: frame_num + 1])
        team_1 = np.sum(history == 1)
        team_2 = np.sum(history == 2)
        total = max(team_1 + team_2, 1)

        overlay = output.copy()
        cv2.rectangle(overlay, (20, 20), (360, 95), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.45, output, 0.55, 0, output)

        cv2.putText(output, f"Team 1 ball control: {100.0 * team_1 / total:.1f}%", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(output, f"Team 2 ball control: {100.0 * team_2 / total:.1f}%", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return output

    @staticmethod
    def _draw_camera_movement(frame, movement):
        output = frame.copy()
        dx, dy = movement
        cv2.putText(output, f"Camera Movement X: {dx:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(output, f"Camera Movement Y: {dy:.2f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return output
    
    def analyze_video(self):
        """Analyze football video"""
        video_frames, fps = self._read_video(self.video_path)
        if not video_frames:
            print(f"Error: Cannot open video {self.video_path}")
            return

        print(f"Loaded {len(video_frames)} frames")

        tracks = self._build_tracks(video_frames)
        tracks["ball"] = self.tracker.interpolate_ball_tracks(tracks["ball"])
        self._add_position_to_tracks(tracks)

        camera_movement_per_frame = self.camera_motion.get_camera_movement_batch(
            video_frames,
            read_from_stub=self.read_from_stub,
            stub_path=self._camera_stub_path(),
        )
        self.camera_motion.add_adjusted_positions_to_tracks(tracks, camera_movement_per_frame)

        if self.perspective.transformation_matrix is not None:
            self.perspective.add_transformed_position_to_tracks(tracks)
        else:
            self._add_fallback_transformed_positions(tracks)

        # Fit across the first N frames so a single occluded opening frame can't skew the clusters.
        if tracks["players"]:
            self.team_assigner.fit_team_colors(video_frames, tracks["players"])
            ta = self.team_assigner
            if ta.kmeans is not None and len(ta.team_colors) == 2:
                centers = list(ta.kmeans.cluster_centers_)
                sep = np.linalg.norm(centers[0] - centers[1])
                c1, c2 = centers
                print(f"Fitted cluster colors (pre-override) — T1: B={c1[0]:.0f} G={c1[1]:.0f} R={c1[2]:.0f} | "
                      f"T2: B={c2[0]:.0f} G={c2[1]:.0f} R={c2[2]:.0f} | "
                      f"separation: {sep:.1f}/442 ({100*sep/442:.0f}% — if low, increase color_sample_frames or set team_color_overrides). "
                      f"Rendered colors may differ if team_color_overrides is set.")
            else:
                print("Team color fitting skipped — not enough distinct player colors in opening frames")

        team_ball_control = []
        for frame_num, player_tracks in enumerate(tracks["players"]):
            for player_id, track in player_tracks.items():
                team_id = self.team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                tracks["players"][frame_num][player_id]["team"] = team_id
                team_color = self.team_assigner.team_colors.get(team_id, np.array(COLORS["player"]))
                tracks["players"][frame_num][player_id]["team_color"] = tuple(int(c) for c in team_color.tolist())

            ball_track = tracks["ball"][frame_num]
            assigned_player = -1
            if ball_track:
                first_ball_id = next(iter(ball_track))
                ball_bbox = ball_track[first_ball_id]["bbox"]
                assigned_player = self.player_ball_assigner.assign_ball_to_player(player_tracks, ball_bbox)

            if assigned_player != -1 and assigned_player in tracks["players"][frame_num]:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player].get("team", 1))
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

        self.speed_calculator.add_speed_and_distance_to_tracks(tracks, frame_window=SPEED_CONFIG["frame_window"])

        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame_tracks = {
                "players": tracks["players"][frame_num],
                "referees": tracks["referees"][frame_num],
                "ball": tracks["ball"][frame_num],
            }
            output = self._draw_tracks(frame, frame_tracks)
            output = self._draw_camera_movement(output, camera_movement_per_frame[frame_num])
            output = self._draw_team_ball_control(output, frame_num, team_ball_control)
            output_frames.append(output)

        output_frames = self.speed_calculator.draw_speed_and_distance(output_frames, tracks)
        self._save_video(output_frames, self.output_path, fps)

        print(f"\nAnalysis complete! Output saved to: {self.output_path}")
        
        # Print statistics
        self._print_statistics(tracks)
    
    def _print_statistics(self, tracks):
        """Print analysis statistics"""
        latest_players = tracks["players"][-1] if tracks["players"] else {}
        print("\n=== Analysis Statistics ===")
        print(f"Tracked players in final frame: {len(latest_players)}")

        printed = set()
        for frame_players in tracks["players"]:
            for player_id, player_info in frame_players.items():
                if player_id in printed:
                    continue
                if "distance" not in player_info:
                    continue
                printed.add(player_id)
                print(f"\nPlayer {player_id}:")
                print(f"  Distance covered: {player_info.get('distance', 0.0):.2f} m")
                print(f"  Current speed: {player_info.get('speed', 0.0):.2f} km/h")


def main():
    video_path = VIDEO_CONFIG["input_path"]

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Place your football video at input/DFL-Scoutingfeed.mp4 and rerun.")
        return

    os.makedirs(os.path.dirname(VIDEO_CONFIG["output_path"]), exist_ok=True)

    analyzer = FootballAnalyzer(video_path=video_path)
    analyzer.analyze_video()


if __name__ == "__main__":
    main()
