import pytest
from src.player_ball_assigner import PlayerBallAssigner


def test_assigns_nearest_player():
    assigner = PlayerBallAssigner(max_player_ball_distance=70)
    players = {
        1: {"bbox": [100, 100, 120, 200]},
        2: {"bbox": [300, 100, 320, 200]},
    }
    assert assigner.assign_ball_to_player(players, [105, 195, 115, 205]) == 1


def test_assigns_closer_of_two_players_in_range():
    assigner = PlayerBallAssigner(max_player_ball_distance=200)
    # player 2's left foot (110, 200) is closer to ball center (112, 203) than
    # player 1's right foot (120, 200)
    players = {
        1: {"bbox": [100, 100, 120, 200]},
        2: {"bbox": [110, 100, 130, 200]},
    }
    assert assigner.assign_ball_to_player(players, [107, 200, 117, 206]) == 2


def test_returns_minus_one_when_no_player_in_range():
    assigner = PlayerBallAssigner(max_player_ball_distance=10)
    players = {1: {"bbox": [100, 100, 120, 200]}}
    assert assigner.assign_ball_to_player(players, [500, 500, 510, 510]) == -1


def test_returns_minus_one_with_no_players():
    assigner = PlayerBallAssigner()
    assert assigner.assign_ball_to_player({}, [100, 100, 110, 110]) == -1


def test_uses_foot_corners_not_center():
    # Foot corners are (x1, y2) and (x2, y2) — bottom of bbox.
    # Ball placed close to the right foot corner (120, 200) but far from bbox center.
    assigner = PlayerBallAssigner(max_player_ball_distance=5)
    players = {1: {"bbox": [100, 50, 120, 200]}}
    assert assigner.assign_ball_to_player(players, [118, 198, 122, 202]) == 1
