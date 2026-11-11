"""Read YOLO-format datasets (images/ + labels/ per split)."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "valid", "test")
_SOURCE_RE = re.compile(r"_mp4-\d+.*$")


@dataclass(frozen=True)
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float


def parse_label_file(path: str | Path) -> list[Box]:
    """Boxes from one YOLO label file; malformed lines are skipped."""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(Box(cls, cx, cy, w, h))
    return boxes


def source_video(filename: str) -> str:
    """Source clip of a Roboflow frame name, e.g. 'adit_mp4-77_jpg.rf.<hash>.jpg' -> 'adit'."""
    return _SOURCE_RE.sub("", Path(filename).name)


def iter_split(root: str | Path, split: str) -> Iterator[tuple[Path, list[Box]]]:
    """Yield (image_path, boxes) for every image in a split, sorted by file name."""
    img_dir = Path(root) / split / "images"
    lbl_dir = Path(root) / split / "labels"
    for img in sorted(img_dir.glob("*.jpg")):
        label = lbl_dir / f"{img.stem}.txt"
        yield img, parse_label_file(label) if label.exists() else []
