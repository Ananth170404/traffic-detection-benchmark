import json

import numpy as np
import pytest

from tdb.eval.coco_eval import evaluate, load_gt, to_coco_results
from tdb.eval.speed import benchmark
from tdb.models.base import Detections

# Two images; image a has a car and a truck, image b has one car. Category ids 1..3.
GT = {
    "images": [
        {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "b.jpg", "width": 640, "height": 480},
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 10, 100, 80],
            "area": 8000,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 3,
            "bbox": [300, 200, 150, 120],
            "area": 18000,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [50, 60, 90, 70],
            "area": 6300,
            "iscrowd": 0,
        },
    ],
    "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}, {"id": 3, "name": "truck"}],
}


@pytest.fixture
def gt(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(GT))
    return load_gt(path)


def oracle(shift: float = 0.0) -> dict[str, Detections]:
    """Detections equal to the ground truth, optionally shifted right by `shift` pixels."""
    out = {}
    for im in GT["images"]:
        anns = [a for a in GT["annotations"] if a["image_id"] == im["id"]]
        boxes = [[x + shift, y, x + w + shift, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        labels = [a["category_id"] - 1 for a in anns]
        out[im["file_name"]] = Detections(boxes, np.ones(len(anns)), labels)
    return out


def test_perfect_predictions_score_one(gt):
    m = evaluate(gt, to_coco_results(gt, oracle()))
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(1.0)
    assert m["per_class"]["car"]["ap50"] == pytest.approx(1.0)
    assert m["per_class"]["car"]["gt_instances"] == 2


def test_class_without_ground_truth_is_nan_and_excluded(gt):
    m = evaluate(gt, to_coco_results(gt, oracle()))
    assert np.isnan(m["per_class"]["bus"]["ap50"])
    assert m["map50"] == pytest.approx(1.0)


def test_shifted_boxes_pass_iou50_but_not_strict_thresholds(gt):
    # A 20 px shift gives IoU 0.64-0.77 on these boxes: all match at 0.5-0.6, none above 0.8.
    m = evaluate(gt, to_coco_results(gt, oracle(shift=20)))
    assert m["map50"] == pytest.approx(1.0)
    assert 0.0 < m["map50_95"] < 0.6


def test_wrong_class_scores_zero(gt):
    wrong = {k: Detections(d.boxes, d.scores, np.full(len(d), 1)) for k, d in oracle().items()}
    m = evaluate(gt, to_coco_results(gt, wrong))
    assert m["map50"] == pytest.approx(0.0)


def test_no_detections(gt):
    assert evaluate(gt, [])["map50"] == 0.0


def test_to_coco_results_converts_boxes_and_labels(gt):
    det = {"a.jpg": Detections([[10, 20, 110, 70]], [0.9], [2])}
    (r,) = to_coco_results(gt, det)
    assert r == {
        "image_id": 1,
        "category_id": 3,
        "bbox": [10.0, 20.0, 100.0, 50.0],
        "score": pytest.approx(0.9),
    }


def test_detections_validates_lengths():
    with pytest.raises(ValueError):
        Detections([[0, 0, 1, 1]], [0.5, 0.6], [0])
    assert len(Detections.empty()) == 0


def test_benchmark_reports_latency():
    images = [np.zeros((4, 4, 3), np.uint8)]
    out = benchmark(lambda im: None, images, warmup=2, runs=10)
    assert out["runs"] == 10 and out["fps"] > 0
    assert out["latency_ms_p50"] <= out["latency_ms_p95"]
