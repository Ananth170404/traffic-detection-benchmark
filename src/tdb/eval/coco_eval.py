"""COCO-style detection metrics shared by every model in the benchmark."""

import contextlib
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tdb.models.base import Detections


def load_gt(ann_file: str | Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(ann_file))


def to_coco_results(gt: COCO, per_image: dict[str, Detections]) -> list[dict]:
    """Convert {file_name: Detections} into COCO results (category_id = label + 1)."""
    file_to_id = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    results = []
    for file_name, det in per_image.items():
        image_id = file_to_id[file_name]
        for (x1, y1, x2, y2), score, label in zip(det.boxes, det.scores, det.labels, strict=True):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return results


def _ap(precision: np.ndarray) -> float:
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else float("nan")


def evaluate(gt: COCO, results: list[dict]) -> dict:
    """mAP@0.5, mAP@0.5:0.95, AP by object size, and per-class AP.

    Classes with no ground truth in the split get NaN AP and are excluded from the means
    (pycocotools behaviour), so e.g. 12-class test mAP is over the classes that occur.
    """
    cat_ids = sorted(gt.getCatIds())
    names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}
    if not results:
        per_class = {names[c]: {"ap50": 0.0, "ap50_95": 0.0} for c in cat_ids}
        return {
            "map50": 0.0,
            "map50_95": 0.0,
            "ap_small": 0.0,
            "ap_medium": 0.0,
            "ap_large": 0.0,
            "ar100": 0.0,
            "per_class": per_class,
            "num_detections": 0,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision: [iou_thresholds(10), recall(101), classes, area_ranges(4), max_dets(3)]
    precision = ev.eval["precision"]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        per_class[names[cat_id]] = {
            "ap50": _ap(precision[0, :, k, 0, -1]),
            "ap50_95": _ap(precision[:, :, k, 0, -1]),
            "gt_instances": len(gt.getAnnIds(catIds=[cat_id])),
        }
    s = ev.stats
    return {
        "map50": float(s[1]),
        "map50_95": float(s[0]),
        "ap_small": float(s[3]),
        "ap_medium": float(s[4]),
        "ap_large": float(s[5]),
        "ar100": float(s[8]),
        "per_class": per_class,
        "num_detections": len(results),
    }
