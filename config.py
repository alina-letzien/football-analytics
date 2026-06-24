"""
Configuration file for football analysis system
"""

# YOLO Configuration
YOLO_CONFIG = {
    "model_path": "yolo26x.pt",  # Options: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "device": "mps",  # "cpu", "mps" (Apple Silicon), or GPU index like "0"
}

# Tracker Configuration (ByteTrack)
TRACKER_CONFIG = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,        # Frames to keep a lost track alive
    "minimum_matching_threshold": 0.8,
    "frame_rate": 30,
}

# Team Assignment Configuration
TEAM_CONFIG = {
    "n_teams": 2,           # Number of teams to identify
    "kmeans_random_state": 42,
}

# Ball Possession Configuration
POSSESSION_CONFIG = {
    "max_player_ball_distance": 70,  # Max pixel distance to assign ball to a player
}

# Perspective Transformation Configuration
FIELD_CONFIG = {
    "field_width": 120.0,   # Meters
    "field_height": 80.0,   # Meters
    "calibration_points": None,  # Will be set during runtime
}

# Speed and Distance Configuration
SPEED_CONFIG = {
    "fps": 30.0,            # Frames per second of video
    "meters_per_pixel": 0.1,  # Conversion factor (requires calibration)
    "frame_window": 5,      # Frames over which speed is averaged
}

# Camera Motion Configuration
CAMERA_CONFIG = {
    "minimum_distance": 2.0,    # Minimum pixel movement to count as camera motion
    # Farnebäck optical flow parameters
    "farneback_pyr_scale": 0.5,
    "farneback_levels": 3,
    "farneback_winsize": 15,
    "farneback_iterations": 3,
    "farneback_poly_n": 5,
    "farneback_poly_sigma": 1.2,
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "input_path": "input/DFL-Scoutingfeed.mp4",
    "output_path": "output/analysis.mp4",
    "output_codec": "mp4v",  # Options: "mp4v", "XVID", "MJPG"
    "read_from_stub": True,
    "stub_dir": "stubs",
}

# Visualization Colors (BGR format)
COLORS = {
    "player": (255, 0, 0),      # Blue (fallback when team color is unknown)
    "ball": (0, 255, 0),        # Green
    "referee": (0, 255, 255),   # Yellow
}
