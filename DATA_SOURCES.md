# Data Sources

Nothing below is stored in this repository; use the download scripts.

## Datasets

| Name | Source | License | Used for |
|---|---|---|---|
| Roboflow-100 Vehicles (vehicles-q0x2v, v2) | https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2 | CC BY 4.0 | Training and evaluating all detectors |

Splits (v2): 2,634 train / 966 valid / 458 test images, 12 classes.

Published RF100 baselines for this dataset (mAP@0.5, from
[roboflow-100-benchmark](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv)):
YOLOv5 0.454, YOLOv7 0.464, YOLOv8 0.472.

The original Colab notebooks used this v2 YOLOv5 export; the HOG+SVM notebook used an earlier
v1 COCO export of the same project. This repo uses v2 only and generates COCO JSON from it.

Roboflow-100 citation:

> Ciaglia, F., Zuppichini, F. S., Guerrie, P., McQuade, M., & Solawetz, J. (2022).
> Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark. arXiv:2211.13523.

## Videos

Only videos whose license allows redistribution of derived clips are used.

| File | Source URL | License | Attribution | Approach in simulated junction |
|---|---|---|---|---|
| _to be added_ | | | | |
