"""Build processed dataset variants (YOLO layout + COCO JSON) from the raw Roboflow export.

Each variant is a self-contained folder:

    <out>/<variant>/
        data.yaml                      # Ultralytics dataset file (path-relative, portable)
        {train,valid,test}/images/     # hard links to the raw images (copies if linking fails)
        {train,valid,test}/labels/     # YOLO labels, class ids remapped for the variant
        annotations/{train,valid,test}.json   # COCO detection format

COCO conventions used here: category ids start at 1 (0 is background for torchvision
detectors), bbox is [x, y, w, h] in pixels, boxes are clipped to the image and boxes with
zero width or height are dropped (and counted).
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from tdb.data.yolo import SPLITS, Box, iter_split


@dataclass
class Variant:
    name: str
    class_names: list[str]
    id_map: dict[int, int]


@dataclass
class SplitStats:
    images: int = 0
    boxes: int = 0
    dropped: int = 0
    per_class: dict[str, int] = field(default_factory=dict)


def yolo_to_coco_bbox(box: Box, width: int, height: int) -> list[float] | None:
    """Normalised (cx, cy, w, h) -> pixel [x, y, w, h], clipped to the image; None if empty."""
    x0 = max(0.0, (box.cx - box.w / 2) * width)
    y0 = max(0.0, (box.cy - box.h / 2) * height)
    x1 = min(float(width), (box.cx + box.w / 2) * width)
    y1 = min(float(height), (box.cy + box.h / 2) * height)
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


def coco_categories(class_names: list[str]) -> list[dict]:
    return [{"id": i + 1, "name": n, "supercategory": "vehicle"} for i, n in enumerate(class_names)]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"Unreadable image: {path}")
    return im.shape[1], im.shape[0]


def build_split(raw_root: Path, out_root: Path, split: str, variant: Variant) -> SplitStats:
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = SplitStats(per_class={n: 0 for n in variant.class_names})
    images, annotations = [], []
    for image_id, (img_path, boxes) in enumerate(iter_split(raw_root, split), start=1):
        width, height = image_size(img_path)
        link_or_copy(img_path, img_out / img_path.name)
        images.append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )

        lines = []
        for box in boxes:
            new_cls = variant.id_map[box.cls]
            bbox = yolo_to_coco_bbox(box, width, height)
            if bbox is None:
                stats.dropped += 1
                continue
            lines.append(f"{new_cls} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}")
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": new_cls + 1,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                }
            )
            stats.per_class[variant.class_names[new_cls]] += 1
        (lbl_out / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    stats.images = len(images)
    stats.boxes = len(annotations)
    coco = {
        "info": {"description": f"vehicles-q0x2v v2, variant {variant.name}, split {split}"},
        "licenses": [{"id": 1, "name": "CC BY 4.0"}],
        "images": images,
        "annotations": annotations,
        "categories": coco_categories(variant.class_names),
    }
    ann_dir = out_root / "annotations"
    ann_dir.mkdir(exist_ok=True)
    (ann_dir / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
    return stats


def write_data_yaml(out_root: Path, class_names: list[str]) -> None:
    # Ultralytics resolves train/val/test relative to `path`, which must be absolute, so the
    # file is machine-specific: re-run prepare_data.py on each machine (local, Kaggle).
    data = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_variant(raw_root: Path, out_dir: Path, variant: Variant) -> dict[str, SplitStats]:
    out_root = out_dir / variant.name
    stats = {split: build_split(raw_root, out_root, split, variant) for split in SPLITS}
    write_data_yaml(out_root, variant.class_names)
    return stats
