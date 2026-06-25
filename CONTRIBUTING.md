# Contributing

Thanks for your interest in contributing! Here's everything you need to get started.

## Ways to contribute

You don't have to write code to contribute. Opening an issue is just as valuable:

- **Found a bug?** Open a [bug report](https://github.com/alina-letzien/football-analytics/issues/new?template=bug_report.md)
- **Have an idea?** Open a [feature request](https://github.com/alina-letzien/football-analytics/issues/new?template=feature_request.md)
- **Want to contribute code?** Follow the steps below

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Place your input video at in the `input` folder or use `input/DFL-Scoutingfeed.mp4`. Output lands at `output/analysis.mp4`.

## Verifying changes

There is no test suite. Verifying a change means locally running the full pipeline:

```bash
./venv/bin/python main.py
```

Then check `output/analysis.mp4` visually.

### Note 
If your change affects detection, tracking, or camera motion, **delete the stubs first** — otherwise the pipeline runs on cached data and your change won't actually execute.

## Things to know before touching code

- **`config.py`** is the single source of all tunable values. All modules receive their config via constructor arguments — except `CameraMotionAnalyzer`, which reads `config.py` via a module-level import. Renaming config keys will break things at runtime.
- **The `tracks` dict** is shared across every pipeline stage and progressively enriched. Changing its key names or structure will silently break all downstream stages.
- **Stubs** in `stubs/` cache detection, tracking, and camera motion results. They speed up re-runs but must be deleted after any logic change that affects those stages.
- **YOLO class names** (`"person"`, `"goalkeeper"`, `"referee"`, `"ball"`, `"sports ball"`) must stay in sync between the model and the detector/tracker routing. Custom-trained models must use matching names.

## Submitting a PR

1. Fork the repo and create a new branch from `main`
2. Make your changes and verify by running the full pipeline
3. Open a PR — the template will guide you through what to include
4. Link the related issue in the PR description if one exists
