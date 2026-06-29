import numpy as np
import pytest

N_FRAMES = 20
PLAYER_ID = 1
BALL_ID = 99


@pytest.fixture
def synthetic_tracks():
    """Minimal tracks dict that mimics detection output — no ML needed."""
    players, ball, referees = [], [], []
    for _ in range(N_FRAMES):
        players.append({PLAYER_ID: {"bbox": [100, 100, 120, 200], "center": (110, 150), "conf": 0.9}})
        ball.append({BALL_ID: {"bbox": [105, 190, 115, 200], "center": (110, 195), "conf": 0.8}})
        referees.append({})
    return {"players": players, "ball": ball, "referees": referees}


@pytest.fixture
def synthetic_frames():
    return [np.zeros((200, 640, 3), dtype=np.uint8) for _ in range(N_FRAMES)]
