"""Download the Roboflow-100 vehicles dataset (YOLOv5 format) into data/vehicles/.

Needs a free Roboflow API key in the ROBOFLOW_API_KEY environment variable.
On Kaggle, attach the dataset as an input instead and skip this script.
"""

import argparse
import os
import shutil
from pathlib import Path

WORKSPACE = "roboflow-100"
PROJECT = "vehicles-q0x2v"
VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/vehicles"))
    parser.add_argument("--format", default="yolov5", help="Roboflow export format")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY (free key from https://app.roboflow.com/settings/api)"
        )

    from roboflow import Roboflow

    args.out.parent.mkdir(parents=True, exist_ok=True)
    project = Roboflow(api_key=api_key).workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(args.format, location=str(args.out))

    for split in ("train", "valid", "test"):
        images = Path(dataset.location) / split / "images"
        n = len(list(images.glob("*.jpg"))) if images.exists() else 0
        print(f"{split:>5}: {n} images")

    readme = Path(dataset.location) / "README.roboflow.txt"
    if readme.exists():
        shutil.copy(readme, args.out / "LICENSE_NOTE.txt")


if __name__ == "__main__":
    main()
