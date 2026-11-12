"""Train an Ultralytics YOLO model (v8 / 11 / 26) with the shared budget in configs/train/yolo.yaml.

    python scripts/train_yolo.py --model yolo11s --data data/processed/v3 --device 0,1

Re-running the same command resumes from runs/yolo/<name>/weights/last.pt if it exists
(Kaggle sessions stop after 12 h). The best checkpoint is copied to weights/<name>.pt,
where <name> is <model>_<variant>[_<suffix>].
"""

import argparse
import shutil
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train" / "yolo.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="key under `models` in yolo.yaml")
    parser.add_argument("--data", type=Path, required=True, help="processed variant folder")
    parser.add_argument("--device", default="0", help="e.g. 0, 0,1 or cpu")
    parser.add_argument("--suffix", default="", help="run name suffix, e.g. clahe")
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--epochs", type=int, default=None, help="override, for smoke tests")
    parser.add_argument("--fraction", type=float, default=None, help="train on a data subset")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(CONFIG.read_text())
    train_args = dict(cfg["train"])
    if args.epochs is not None:
        train_args["epochs"] = args.epochs
    if args.fraction is not None:
        train_args["fraction"] = args.fraction
    name = "_".join(p for p in (args.model, args.data.name, args.suffix) if p)

    last = args.project / name / "weights" / "last.pt"
    if last.exists():
        print(f"Resuming {name} from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(cfg["models"][args.model])
        model.train(
            data=str((args.data / "data.yaml").resolve()),
            device=args.device,
            project=str(args.project.resolve()),
            name=name,
            exist_ok=True,
            **train_args,
        )

    best = args.project / name / "weights" / "best.pt"
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.weights_dir / f"{name}.pt")
    print(f"Saved {args.weights_dir / f'{name}.pt'}")


if __name__ == "__main__":
    main()
