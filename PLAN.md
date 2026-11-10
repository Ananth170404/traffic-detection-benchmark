# Project Plan

Benchmark classical, CNN, two-stage and one-stage detectors on a traffic dataset under one
evaluation protocol, then use the best detector + tracking to count vehicles and drive
adaptive signal timing (Webster).

## Compute

| Task | Where |
|---|---|
| Training deep models (Faster R-CNN, RetinaNet, YOLO) | Kaggle T4 (YOLO uses 2x T4 via `device=0,1`) |
| FPS benchmarks for every model | Kaggle, single T4, always |
| Development, 1-epoch smoke tests, HOG+SVM, tracking, demo | Local RTX 3050 |

Kaggle notebooks clone this repo, `pip install -e .`, read the dataset from a Kaggle Dataset
input, and write weights + a results JSON. Results JSON is committed to `results/`; weights go
to GitHub Releases. All training scripts checkpoint and resume (12 h session limit).

## Data policy

- No datasets or videos in git. `scripts/download_data.py` fetches from Roboflow.
- Every dataset/video is listed in `DATA_SOURCES.md` with URL, license and attribution.
- Splits: Roboflow train/valid/test. Tune on valid, report on test only.

## Phase 1 — Data and a fair evaluation setup

1. Data exploration notebook: class counts, box sizes, which source videos (filename
   prefixes `adit_`, `aditganteng_`, `pagi_`, `malam_`, `siang_`) use which label scheme.
2. Class merge 12 → car / bus / truck via `configs/classes.yaml`, final mapping decided by
   step 1. Report both 12-class and merged results.
3. Converters: YOLO labels → COCO JSON; merged-class dataset copy.
4. One evaluation harness (`src/tdb/eval/`): COCO mAP@0.5, mAP@0.5:0.95, per-class AP;
   FPS (batch 1, 640 px, T4, incl. post-processing, after warmup, FP32 and FP16);
   params, FLOPs, weight size.

## Phase 2 — Model ladder

| # | Model | Purpose | Owner |
|---|---|---|---|
| 1 | HOG + SVM sliding window + NMS | Classical baseline | Ananth |
| 2 | Simple CNN single-box regression | Why naive regression fails | Ananth |
| 3 | Faster R-CNN ResNet50-FPN, all classes | Two-stage | Claude |
| 4 | RetinaNet | One-stage, non-YOLO | Claude |
| 5 | YOLOv5s | Original model | Claude |
| 6 | YOLOv8s, YOLO11s | Modern YOLO | Claude |

All at 640 px; models 3–6 COCO-pretrained; fixed epoch budget per model documented in its
config. Preprocessing ablation (none / Gaussian blur / CLAHE) on YOLO11s and Faster R-CNN.

Outputs: results table, speed-vs-accuracy scatter, per-class AP charts, written analysis.

## Phase 3 — Counting unique vehicles

Best detector + ByteTrack; per-video YAML with counting line and lane polygons; vehicles/h
per class and PCU/h per approach (car 1.0, bus/truck ~3.0); manual-count validation on 2 clips
with reported counting error; annotated output videos.

## Phase 4 — Signal timing (owner: Ananth)

Each video is treated as one approach of a simulated 4-way junction (stated openly).
Webster: y = q/s (s = 1800 PCU/h/lane), lost time L, C₀ = (1.5L + 5)/(1 − Y), green split by
y, min/max green. Controllers: fixed-time, one-shot Webster, rolling Webster. Compared with
Webster's delay formula. Timings overlaid on output video.

## Phase 5 — Presentation and quality

README (diagram, results, GIF, reproduce commands, limitations); Gradio demo on HF Spaces
(YOLO11s ONNX, CPU); pytest + ruff in GitHub Actions; `docs/model_card.md`; short write-up.

## Working agreement

- Ananth implements HOG+SVM, simple CNN and Webster; Claude writes interfaces + tests first.
- Claude writes the rest, using Ananth's Colab notebooks as reference and fixing known bugs.
- Each phase ends with a commit and a short results note.

## Known issues in the original notebooks (fixed in this repo)

- "Blur" YOLO runs used `data.yaml` (original images), so the blur comparison never ran.
- CNN / Faster R-CNN notebook trained and evaluated on the same split.
- Faster R-CNN "accuracy" was matched boxes per image, not a percentage.
- IoU > 0.01 thresholds; 13-class Faster R-CNN crashed on empty images and used normalised
  boxes.
- Webster notebook split a fixed 60 s cycle by flow and counted the same vehicle every frame.
- HOG/SVM: scaler fit before split; feature/label order not guaranteed to match; "batch"
  SVM refit from scratch each batch.
