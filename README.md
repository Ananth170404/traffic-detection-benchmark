# Traffic Detection Benchmark and Adaptive Signal Timing

> Work in progress. See [PLAN.md](PLAN.md) for scope and status.

Compares a classical detector (HOG + SVM), a simple CNN, Faster R-CNN, RetinaNet and several
YOLO versions on the same traffic dataset, with one evaluation protocol (COCO mAP, FPS on a
T4, model size). The best detector is combined with ByteTrack to count vehicles per approach,
and the counts drive Webster's method for signal timing.

## Status

| Phase | Status |
|---|---|
| 1. Data and evaluation setup | In progress |
| 2. Model benchmark | Not started |
| 3. Vehicle counting | Not started |
| 4. Signal timing | Not started |
| 5. Demo and write-up | Not started |

## Setup

```bash
pip install -e ".[train,dev]"
export ROBOFLOW_API_KEY=...        # free key from roboflow.com
python scripts/download_data.py    # -> data/vehicles/
pytest
```

## Repository layout

```
configs/    class mapping, model and video configs
scripts/    data download, conversion, training, evaluation
src/tdb/    data, models, eval, tracking, signal packages
notebooks/  exploration and Kaggle training notebooks
results/    committed metrics (JSON) and plots
app/        Gradio demo
tests/      unit tests
```

## Data

See [DATA_SOURCES.md](DATA_SOURCES.md). Dataset: Roboflow-100 Vehicles (CC BY 4.0).

## License

Code: MIT. Data and videos keep their original licenses.
