## Related issue
<!-- Closes #<issue-number> -->

## What does this PR do?
<!-- Describe the change and the motivation behind it. -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Docs / config only

## Which pipeline stage does this affect?
- [ ] Detection (`YOLODetector`)
- [ ] Tracking (`ByteTrackTracker`)
- [ ] Camera motion (`CameraMotionAnalyzer`)
- [ ] Position / perspective (`PerspectiveTransformer`)
- [ ] Team assignment (`TeamAssigner`)
- [ ] Ball possession (`PlayerBallAssigner`)
- [ ] Speed / distance (`SpeedDistanceCalculator`)
- [ ] Rendering / output
- [ ] Config (`config.py`)
- [ ] Training (`train.py`)

## Breaking changes
<!-- Does this change any of the following? If yes, describe the impact. -->
- [ ] `tracks` dict keys or structure (will silently break all downstream stages)
- [ ] `config.py` key names or dict structure (modules import directly — renames break at runtime)
- [ ] Stub format (existing stubs in `stubs/` must be deleted before re-running)
- [ ] YOLO class names (must stay in sync with model: `"person"`, `"goalkeeper"`, `"referee"`, `"ball"`, `"sports ball"`)

## Changes to `config.py`
<!-- List any tunable values added, removed, or changed. None if not applicable. -->

## How was this tested?
<!-- No test suite — describe manual verification steps. -->

- [ ] Ran `python main.py` end-to-end
- [ ] Checked `output/analysis.mp4` visually for regressions
- [ ] Deleted stubs before re-running (required if detection or camera logic changed)
- [ ] Tested on Apple Silicon MPS / CPU / CUDA (circle which)

## Screenshots / output comparison
<!-- Before/after frames or metrics if the change affects visuals. -->

## Notes for reviewer
<!-- Non-obvious implementation details, known limitations, or follow-up work. -->
