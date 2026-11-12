"""Evaluate any detector on a dataset split with the shared protocol.

    python scripts/evaluate.py --detector tdb.models.yolo:UltralyticsDetector \\
        --weights weights/yolo11s_v3.pt --data data/processed/v3 --split test --name yolo11s_v3

Writes results/<name>.json (metrics, speed, model size, settings) and
results/predictions/<name>_<split>.json (COCO results, for re-scoring without re-running).

The detector class is loaded from "module:Class" and constructed as
Class(weights=..., class_names=..., **opts); --opt key=value adds keyword arguments.
"""

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from tdb.eval.coco_eval import evaluate, load_gt, to_coco_results
from tdb.eval.model_stats import count_flops, count_parameters, file_size_mb
from tdb.eval.speed import benchmark


def load_detector(spec: str, **kwargs):
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise SystemExit(f"--detector must look like 'package.module:Class', got {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**kwargs)


def parse_opts(pairs: list[str]) -> dict:
    opts = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        opts[key] = yaml.safe_load(value)  # "0.25" -> float, "true" -> bool, "x" -> str
    return opts


def model_info(detector) -> dict:
    """Optional hooks: detector.module (torch nn.Module), detector.example_input(), weights."""
    info = {"weights_mb": file_size_mb(getattr(detector, "weights", None))}
    module = getattr(detector, "module", None)
    if module is not None:
        info["params"] = count_parameters(module)
        if hasattr(detector, "example_input"):
            info["flops"] = count_flops(module, detector.example_input())
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, help="module:Class")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data", type=Path, required=True, help="processed variant folder")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--name", required=True, help="run name used for output files")
    parser.add_argument("--opt", action="append", default=[], help="extra key=value kwargs")
    parser.add_argument("--speed-runs", type=int, default=200, help="0 to skip speed test")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    class_names = yaml.safe_load((args.data / "data.yaml").read_text())["names"]
    opts = parse_opts(args.opt)
    detector = load_detector(args.detector, weights=args.weights, class_names=class_names, **opts)

    gt = load_gt(args.data / "annotations" / f"{args.split}.json")
    img_dir = args.data / args.split / "images"
    per_image = {}
    for im in tqdm(gt.dataset["images"], desc=f"{args.name} on {args.split}"):
        image = cv2.imread(str(img_dir / im["file_name"]))
        per_image[im["file_name"]] = detector.predict(image)
    results = to_coco_results(gt, per_image)
    metrics = evaluate(gt, results)

    speed = None
    if args.speed_runs > 0:
        sample = [cv2.imread(str(img_dir / im["file_name"])) for im in gt.dataset["images"][:50]]
        speed = benchmark(detector.predict, sample, runs=args.speed_runs)

    report = {
        "name": args.name,
        "detector": args.detector,
        "weights": args.weights,
        "options": opts,
        "data": args.data.name,
        "split": args.split,
        "classes": class_names,
        "metrics": metrics,
        "speed": speed,
        "model": model_info(detector),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions").mkdir(exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "predictions" / f"{args.name}_{args.split}.json").write_text(
        json.dumps(results), encoding="utf-8"
    )

    print(f"\n{args.name} [{args.data.name}/{args.split}]")
    print(f"  mAP@0.5 {metrics['map50']:.3f}   mAP@0.5:0.95 {metrics['map50_95']:.3f}")
    for cls, m in metrics["per_class"].items():
        print(f"    {cls:>12}: AP50 {m['ap50']:.3f}  AP50-95 {m['ap50_95']:.3f}")
    if speed:
        print(f"  {speed['latency_ms_mean']} ms/img ({speed['fps']} FPS) on {speed['device']}")


if __name__ == "__main__":
    main()
