# Football Analytics Pipeline

Computer vision football analysis pipeline with YOLO, ByteTrack, OpenCV, and Python.

## About the project

This project analyzes a football video and produces an annotated output video with:

- tracked players, referees, and ball
- team assignment (based on jersey colors)
- ball possession per frame
- team ball-control percentage
- player speed and distance estimates


### Context

The project was created to learn and apply practical computer vision concepts in a real-world scenario.
It focuses on building an end-to-end pipeline, integrating model and tracker into a cohesive video analytics workflow, with the goal of providing an analytics baseline that can be extended and retrained on custom data.

### How It Works (High Level)

1. Detect objects in each frame with YOLO.
2. Track detected objects over time with ByteTrack.
3. Estimate camera movement and adjust object positions.
4. Assign players to teams using jersey color clustering (KMeans).
5. Associate ball possession with nearest player.
6. Estimate speed and distance from tracked movement.
7. Render overlays and export the final analysis video.

## Features

- YOLO detection for players, referees, and ball
- ByteTrack multi-object tracking with persistent track IDs
- team assignment from jersey color (KMeans)
- ball interpolation across missed detections
- ball possession assignment to nearest player
- team ball-control overlay in output video
- camera motion estimation and position adjustment
- perspective-transform integration (with fallback scaling)
- player speed and distance estimation
- optional stub caching for faster repeated runs

## Project Structure

```text
football-analytics/
├── input/                        # Input videos (expected: DFL-Scoutingfeed.mp4)
├── output/                       # Annotated output videos
├── stubs/                        # Cached intermediate files from analysis
├── datasets/                     # Local training datasets
├── src/
│   ├── yolo_detector.py
│   ├── tracker.py                # ByteTrackTracker
│   ├── team_assigner.py
│   ├── player_ball_assigner.py
│   ├── camera_motion.py
│   ├── perspective_transformer.py
│   └── speed_distance_calculator.py
├── tests/
│   ├── conftest.py               # Shared fixtures (synthetic tracks and frames)
│   ├── test_player_ball_assigner.py
│   ├── test_speed_distance.py
│   ├── test_pipeline_integration.py
│   └── test_smoke.py             # Full pipeline smoke test (marked slow)
├── config.py
├── main.py                       # Main analysis entrypoint
├── train.py                      # Local YOLO training entrypoint
├── pytest.ini
└── requirements.txt
```

## Pipeline Architecture

- `main.py`: orchestrates full analysis run
- `src/yolo_detector.py`: YOLO inference and object parsing
- `src/tracker.py`: ByteTrack integration and track state
- `src/team_assigner.py`: team clustering and team ID assignment
- `src/player_ball_assigner.py`: nearest-player possession assignment
- `src/camera_motion.py`: optical-flow camera motion estimation
- `src/perspective_transformer.py`: coordinate transformation utilities
- `src/speed_distance_calculator.py`: speed/distance computation and overlays

## Installation

### Prerequisites

- Python 3.13+

### Setup

Run the following commands within a terminal:

```bash
git clone https://github.com/alina-letzien/football-analytics.git
cd football-analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Analysis

1. Put an input video inside the `input` folder (or use the default `input/DFL-Scoutingfeed.mp4`).
2. Open a terminal and run `./venv/bin/python main.py` inside the project root.
    - The YOLO Model is downloaded automatically on first run if not present locally.

### Output and generated files

- `output/analysis.mp4`
- `stubs/track_stubs.pkl`
- `stubs/camera_movement_stubs.pkl`

## Testing

The test suite has three layers with different speed and dependency requirements.

### Fast tests — unit and integration (no ML, no video)

```bash
pytest
```

Runs 10 tests in ~5 seconds. No YOLO model or input video required.

| Layer | File | What it covers |
|---|---|---|
| Unit | `tests/test_player_ball_assigner.py` | Ball-to-player assignment logic, foot-corner distance |
| Unit | `tests/test_speed_distance.py` | Speed formula, distance accumulation, Pythagorean math |
| Integration | `tests/test_pipeline_integration.py` | All post-detection stages on synthetic data — position enrichment, camera motion, team assignment, possession, speed/distance |

### Smoke test — full pipeline (requires input video and YOLO model)

```bash
pytest -m slow
```

Runs `main.py` end-to-end and asserts `output/analysis.mp4` is written. Skipped automatically if `input/DFL-Scoutingfeed.mp4` is not present.

> **Before running the smoke test after a dependency bump:** delete `stubs/track_stubs.pkl` and `stubs/camera_movement_stubs.pkl` first. Without this, the pipeline loads cached detections and never calls YOLO or torch — the updated packages are not actually exercised.

```bash
rm stubs/*.pkl
pytest -m slow
```

### Run only fast tests explicitly

```bash
pytest -m "not slow"
```

## Example Outcome

After a full run, the output video includes:

- bounding boxes and track IDs
- team labels and ball possession markers
- camera motion info
- team ball-control percentages
- speed and distance text near tracked players

## Configuration

All tunable settings are in `config.py` — it is the single place to adjust the pipeline without touching source code:

- `YOLO_CONFIG` — model path, device, confidence and IoU thresholds
- `TRACKER_CONFIG` — ByteTrack parameters
- `TEAM_CONFIG` — number of teams, KMeans random state
- `POSSESSION_CONFIG` — max distance to assign ball to a player
- `FIELD_CONFIG` — field dimensions in meters
- `SPEED_CONFIG` — FPS, meters-per-pixel scale, speed averaging window
- `CAMERA_CONFIG` — motion threshold, Farnebäck optical flow parameters
- `VIDEO_CONFIG` — input/output paths, codec, stub caching toggle
- `COLORS` — bounding box colors for players, ball, and referees (BGR format)

### Note
If logic changed, delete files in `stubs/` and rerun to recompute cached data.

## Training

The pipeline works out of the box with the base `yolo26x.pt` from Ultralytics — a generic model that detects `"person"`, `"sports ball"`, and similar classes. Training your own model means fine-tuning on a football-specific dataset where objects are labeled as `player`, `goalkeeper`, `referee`, and `ball`. This improves detection accuracy, especially for distinguishing referees from players and the ball from the background.

You only need to do this if the base model's detection quality is not good enough for your use case.

If you do want to train:

1. Export a Roboflow football dataset in Ultralytics/YOLO format and drop it into the `datasets/` folder.
2. Open a terminal and run the following command from the project root:

```bash
./venv/bin/python train.py \
  --data datasets/football-players-detection.v2i.yolo26/data.yaml \
  --model yolo26x.pt \
  --epochs 100 \
  --imgsz 640 \
  --device mps
```
Best checkpoint is written to `runs/train/football_detector/weights/best.pt`.

3. After training, set `YOLO_CONFIG["model_path"]` in `config.py` to `"runs/train/football_detector/weights/best.pt"`.

### Notes

- on Apple Silicon, use `--device mps` for training and set `"device": "mps"` in `YOLO_CONFIG` in `config.py` for inference
- for CPU runs, use `--device cpu` for training and `"device": "cpu"` in `YOLO_CONFIG` for inference

## Current Scope and Limitations

- optimized for single-camera broadcast-style footage
- all frames are loaded into memory before processing — not suitable for long videos
- team assignment uses KMeans on the first frame only — can misclassify players with similar jersey colors
- track ID resets (after occlusion or track loss) break per-player speed and distance accumulation
- speed and distance estimates are approximations — accurate values require manual perspective calibration via `PerspectiveTransformer.set_calibration_points()`
- heavy occlusions can reduce tracking quality
- no structured data export — stats are printed to `stdout` only
- coaches and bench staff are detected as players — the base model has no concept of in-play vs. out-of-play persons
- player detection quality depends on model — the base model can miss players under occlusion or crowding; a custom-trained model improves this (see Training section)
- no tactical modules yet (formations, pass maps, xG, heatmaps)

## Roadmap Ideas

- export per-player stats (distance, speed, possession) to CSV
- add player heatmaps per player and per team
- add pass and shot event detection
- add formation detection from player position clusters
- auto field calibration via pitch line detection (no manual calibration points)
- add player re-identification across track ID resets

## Resources

### Detection & Tracking

- [Ultralytics YOLO — Docs](https://docs.ultralytics.com/) — full reference for model loading, inference, training, and export; covers all YOLO variants
- [Ultralytics YOLO — GitHub](https://github.com/ultralytics/ultralytics) — source, issues, and model release notes; useful when upgrading `ultralytics`
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864) — original multi-object tracking algorithm used under the hood via supervision
- [Supervision — Docs](https://supervision.roboflow.com/) — API reference for `ByteTrack`, `Detections`, annotators, and all utilities wrapping the tracker
- [Supervision — GitHub](https://github.com/roboflow/supervision) — changelog and migration guides; check here after supervision version bumps

### Video & Image Processing

- [OpenCV — Docs](https://docs.opencv.org/) — reference for video I/O, drawing, and the optical flow used in `CameraMotionAnalyzer`

### Machine Learning

- [PyTorch — Docs](https://pytorch.org/docs/stable/) — framework used by Ultralytics for inference and training
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — team color clustering used in `TeamAssigner`

### Datasets

- [Roboflow Universe](https://universe.roboflow.com/) — browse and export football datasets in YOLO format; search "football players detection" to find the dataset referenced in `train.py`
- [Roboflow Export Guide](https://docs.roboflow.com/exporting-data) — how to export a dataset in Ultralytics/YOLO format compatible with `train.py`

## License

MIT License. See `LICENSE`.
