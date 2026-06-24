import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict


class TeamAssigner:
    """Assign players to teams based on t-shirt color using KMeans"""
    
    def __init__(self, n_teams: int = 2, random_state: int = 42):
        self.n_teams = n_teams
        self.random_state = random_state
        self.team_colors = {}
        self.player_team_assignment = {}
        self.kmeans = None
    
    def get_player_color(self, frame: np.ndarray, bbox: Dict) -> np.ndarray:
        """
        Extract player t-shirt color from bounding box
        
        Args:
            frame: Input frame
            bbox: Bounding box dictionary with x1, y1, x2, y2
            
        Returns:
            Average color of player's t-shirt (BGR)
        """
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        # Get the upper half of bounding box (t-shirt area)
        player_region = frame[y1:int((y1 + y2) * 0.5), x1:x2]
        
        if player_region.size == 0:
            return np.array([0, 0, 0])
        
        # Reshape to 2D array of pixels
        pixels = player_region.reshape(-1, 3)
        
        # Calculate average color
        avg_color = np.mean(pixels, axis=0)
        
        return avg_color
    
    def assign_team_color(self, frame: np.ndarray, player_tracks: Dict) -> None:
        """Fit team color clusters from players in the first frame"""
        if len(player_tracks) < 2:
            return

        player_colors = []
        for _, player_track in player_tracks.items():
            x1, y1, x2, y2 = player_track["bbox"]
            bbox = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            player_colors.append(self.get_player_color(frame, bbox))

        self.kmeans = KMeans(n_clusters=min(self.n_teams, len(player_colors)), random_state=self.random_state, n_init=10)
        self.kmeans.fit(np.array(player_colors))

        for cluster_idx, center in enumerate(self.kmeans.cluster_centers_):
            self.team_colors[cluster_idx + 1] = center

    def get_player_team(self, frame: np.ndarray, player_bbox: List[float], player_id: int) -> int:
        """Get persistent team id for player track id"""
        if player_id in self.player_team_assignment:
            return self.player_team_assignment[player_id]

        if self.kmeans is None:
            return 1

        bbox = {
            "x1": int(player_bbox[0]),
            "y1": int(player_bbox[1]),
            "x2": int(player_bbox[2]),
            "y2": int(player_bbox[3]),
        }
        player_color = self.get_player_color(frame, bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        self.player_team_assignment[player_id] = team_id
        return team_id
