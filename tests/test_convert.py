import json

import cv2
import numpy as np
import pytest
import yaml

from tdb.data.convert import Variant, build_variant, coco_categories, yolo_to_coco_bbox
from tdb.data.yolo import Box


def test_bbox_conversion():
    assert yolo_to_coco_bbox(Box(0, 0.5, 0.5, 0.25, 0.5), 640, 480) == [240.0, 120.0, 160.0, 240.0]


def test_bbox_is_clipped_to_image():
    # Box hanging off the left/top edge.
    assert yolo_to_coco_bbox(Box(0, 0.05, 0.05, 0.2, 0.2), 100, 100) == [0.0, 0.0, 15.0, 15.0]


def test_degenerate_bbox_returns_none():
    assert yolo_to_coco_bbox(Box(0, 0.5, 0.5, 0.0, 0.1), 640, 480) is None
    assert yolo_to_coco_bbox(Box(0, 1.2, 0.5, 0.1, 0.1), 640, 480) is None


def test_categories_start_at_one():
    assert coco_categories(["car", "bus"]) == [
        {"id": 1, "name": "car", "supercategory": "vehicle"},
        {"id": 2, "name": "bus", "supercategory": "vehicle"},
    ]


@pytest.fixture
def raw_dataset(tmp_path):
    root = tmp_path / "raw"
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        cv2.imwrite(str(root / split / "images" / "a.jpg"), img)
        cv2.imwrite(str(root / split / "images" / "b.jpg"), img)
        # a: one class-2 box and one degenerate box; b: empty label file
        (root / split / "labels" / "a.txt").write_text("2 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.0 0.2\n")
        (root / split / "labels" / "b.txt").write_text("")
    return root


def test_build_variant_end_to_end(raw_dataset, tmp_path):
    variant = Variant("v2", ["car", "bus"], {0: 0, 1: 0, 2: 1})
    stats = build_variant(raw_dataset, tmp_path / "out", variant)

    out = tmp_path / "out" / "v2"
    assert stats["train"].images == 2
    assert stats["train"].boxes == 1
    assert stats["train"].dropped == 1
    assert stats["train"].per_class == {"car": 0, "bus": 1}

    assert (out / "train" / "images" / "a.jpg").exists()
    assert (out / "train" / "labels" / "a.txt").read_text().split()[0] == "1"
    assert (out / "train" / "labels" / "b.txt").read_text() == ""

    coco = json.loads((out / "annotations" / "train.json").read_text())
    assert [im["file_name"] for im in coco["images"]] == ["a.jpg", "b.jpg"]
    assert coco["images"][0]["width"] == 64 and coco["images"][0]["height"] == 48
    (ann,) = coco["annotations"]
    assert ann["category_id"] == 2 and ann["bbox"] == [16.0, 12.0, 32.0, 24.0]

    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == ["car", "bus"] and data["val"] == "valid/images"


def test_build_variant_is_rerunnable(raw_dataset, tmp_path):
    variant = Variant("v2", ["car", "bus"], {0: 0, 1: 0, 2: 1})
    build_variant(raw_dataset, tmp_path / "out", variant)
    stats = build_variant(raw_dataset, tmp_path / "out", variant)
    assert stats["test"].boxes == 1
