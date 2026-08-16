"""
Smoke test — runs the full pipeline end-to-end.
Requires the input video and YOLO model; skipped automatically if they are absent.
Run explicitly with: pytest -m slow
"""
import json
import os
import subprocess
import sys

import pytest

VIDEO_PATH = "input/DFL-Scoutingfeed.mp4"
OUTPUT_PATH = "output/analysis.mp4"
SUMMARY_PATH = "output/analysis_summary.json"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tolerance bands calibrated via calibrate_smoke_thresholds.py against a
# baseline run of input/DFL-Scoutingfeed.mp4. Re-run the script and update
# these after any intentional change to detection/tracking behavior (model
# version, confidence thresholds, dependency bumps to
# ultralytics/torch/supervision).
AVG_PLAYERS_LOW = 10.2
AVG_PLAYERS_HIGH = 16.7
AVG_REFEREES_LOW = 0.0
# Floored at 1.0 so a baseline of 0.0 referees still leaves headroom —
# otherwise the band collapses to [0.0, 0.0] and asserts referees are
# always exactly zero, failing CI the moment detection improves.
AVG_REFEREES_HIGH = 1.0
BALL_DETECTION_RATE_MIN = 0.15
PLAYER_TRACK_ID_COUNT_LOW = 25
PLAYER_TRACK_ID_COUNT_HIGH = 46
# Relative to baseline, not a hardcoded physiological limit — this
# pipeline has no perspective calibration configured
# (FIELD_CONFIG["calibration_points"] is None), so absolute speed
# values are known-approximate (see README limitations). This catches
# a further regression in speed calculation, not physiological
# implausibility.
SPEED_KMH_MAX_CEILING = 121.8
TEAM_SPLIT_MAX_SKEW = 0.30  # max fractional imbalance between team_split[0] and team_split[1]


@pytest.mark.slow
def test_full_pipeline_completes():
    if not os.path.exists(os.path.join(PROJECT_ROOT, VIDEO_PATH)):
        pytest.skip("input video not found")

    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        timeout=1800,  # generous margin for CPU-only CI runners running the full 750-frame video
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert os.path.exists(os.path.join(PROJECT_ROOT, OUTPUT_PATH))

    summary_path = os.path.join(PROJECT_ROOT, SUMMARY_PATH)
    assert os.path.exists(summary_path), "main.py did not write analysis_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    assert AVG_PLAYERS_LOW <= summary["avg_players_per_frame"] <= AVG_PLAYERS_HIGH, summary
    assert AVG_REFEREES_LOW <= summary["avg_referees_per_frame"] <= AVG_REFEREES_HIGH, summary
    assert summary["ball_detection_rate"] >= BALL_DETECTION_RATE_MIN, summary
    assert PLAYER_TRACK_ID_COUNT_LOW <= summary["player_track_id_count"] <= PLAYER_TRACK_ID_COUNT_HIGH, summary
    assert 0.0 <= summary["speed_kmh_max"] <= SPEED_KMH_MAX_CEILING, summary

    team_1, team_2 = summary["team_split"]
    total = max(team_1 + team_2, 1)
    skew = abs(team_1 - team_2) / total
    assert skew <= TEAM_SPLIT_MAX_SKEW, f"team split too imbalanced: {summary['team_split']}"
