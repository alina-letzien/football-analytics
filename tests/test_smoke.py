"""
Smoke test — runs the full pipeline end-to-end.
Requires the input video and YOLO model; skipped automatically if they are absent.
Run explicitly with: pytest -m slow
"""
import os
import subprocess
import sys

import pytest

VIDEO_PATH = "input/DFL-Scoutingfeed.mp4"
OUTPUT_PATH = "output/analysis.mp4"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_full_pipeline_completes():
    if not os.path.exists(os.path.join(PROJECT_ROOT, VIDEO_PATH)):
        pytest.skip("input video not found")

    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        timeout=600,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert os.path.exists(os.path.join(PROJECT_ROOT, OUTPUT_PATH))
