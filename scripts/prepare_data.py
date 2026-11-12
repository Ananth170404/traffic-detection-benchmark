"""Build the 12-class and merged 3-class dataset variants from the raw Roboflow export.

    python scripts/prepare_data.py          # local: data/vehicles -> data/processed
    python scripts/prepare_data.py --raw /kaggle/input/vehicles --out /kaggle/working/data

Writes YOLO labels, COCO JSON and data.yaml per variant (see tdb.data.convert).
"""

import argparse
from pathlib import Path

from tdb.data.classes import ClassConfig
from tdb.data.convert import Variant, build_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/vehicles"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--variants", nargs="+", default=["v12", "v3"], choices=["v12", "v3"])
    args = parser.parse_args()

    cfg = ClassConfig.load()
    variants = {
        "v12": Variant("v12", cfg.original, {i: i for i in range(len(cfg.original))}),
        "v3": Variant("v3", cfg.merged, cfg.id_map()),
    }
    for name in args.variants:
        stats = build_variant(args.raw, args.out, variants[name])
        print(f"\n{name} -> {args.out / name}")
        for split, s in stats.items():
            print(f"  {split:>5}: {s.images} images, {s.boxes} boxes, {s.dropped} dropped")
        names = variants[name].class_names
        totals = {c: sum(s.per_class[c] for s in stats.values()) for c in names}
        print("  per class:", totals)


if __name__ == "__main__":
    main()
