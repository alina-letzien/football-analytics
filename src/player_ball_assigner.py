from typing import Dict, Tuple


class PlayerBallAssigner:
    """Assign the ball to the nearest player foot position"""

    def __init__(self, max_player_ball_distance: float = 70.0):
        self.max_player_ball_distance = max_player_ball_distance

    @staticmethod
    def _center_of_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    def assign_ball_to_player(self, players: Dict, ball_bbox: Tuple[float, float, float, float]) -> int:
        """Return the closest player track id, or -1 if no player is close enough"""
        ball_center = self._center_of_bbox(ball_bbox)

        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            x1, y1, x2, y2 = player["bbox"]

            left_foot = (x1, y2)
            right_foot = (x2, y2)

            distance = min(
                self._distance(left_foot, ball_center),
                self._distance(right_foot, ball_center),
            )

            if distance <= self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
