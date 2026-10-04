# Road Scene YOLO

Real-time road scene object detection and scene understanding using YOLOv8.

This repository contains the road-scene detection project, scripts, tests, notebook, sample assets, and CI configuration.

## Project structure

- `road_scene/` — detection, scene understanding, schemas, and video utilities
- `scripts/` — training, evaluation, image/video inference, and asset generation
- `assets/` — sample images and generated outputs
- `notebooks/` — exploratory notebook
- `tests/` — unit tests
- `app.py` — application entry point

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training

The training script is in `scripts/train.py`. It is designed for YOLOv8 fine-tuning and can be configured for a suitable road-scene dataset.

## License

See `LICENSE`.
