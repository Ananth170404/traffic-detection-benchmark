"""Train YOLOv5s with the original ultralytics/yolov5 repo, using the shared budget.

    python scripts/train_yolov5.py --data data/processed/v3 --gpus 2

Clones the repo into ./yolov5 on first use and records its commit in the run folder.
Multi-GPU uses torch.distributed.run, as the yolov5 docs recommend. Re-running resumes
from runs/yolov5/<name>/weights/last.pt. Best weights are copied to weights/<name>.pt.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train" / "yolo.yaml"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="processed variant folder")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--repo-dir", type=Path, default=Path("yolov5"))
    parser.add_argument("--project", type=Path, default=Path("runs/yolov5"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--epochs", type=int, default=None, help="override, for smoke tests")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    t = cfg["train"]
    epochs = args.epochs if args.epochs is not None else t["epochs"]
    name = f"yolov5s_{args.data.name}"
    project = args.project.resolve()

    if not args.repo_dir.exists():
        run(["git", "clone", "--depth", "1", cfg["yolov5"]["repo"], str(args.repo_dir)])
        run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=args.repo_dir,
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_dir, check=True, capture_output=True, text=True
    ).stdout.strip()

    launcher = [sys.executable]
    if args.gpus > 1:
        launcher += ["-m", "torch.distributed.run", "--nproc_per_node", str(args.gpus)]
    last = project / name / "weights" / "last.pt"
    if last.exists():
        cmd = launcher + ["train.py", "--resume", str(last)]
    else:
        cmd = launcher + [
            "train.py",
            "--data",
            str((args.data / "data.yaml").resolve()),
            "--weights",
            cfg["yolov5"]["weights"],
            "--img",
            str(t["imgsz"]),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(t["batch"]),
            "--patience",
            str(t["patience"]),
            "--seed",
            str(t["seed"]),
            "--workers",
            str(t["workers"]),
            "--device",
            ",".join(str(i) for i in range(args.gpus)),
            "--project",
            str(project),
            "--name",
            name,
            "--exist-ok",
        ]
    run(cmd, cwd=args.repo_dir)

    (project / name / "yolov5_commit.txt").write_text(commit + "\n")
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project / name / "weights" / "best.pt", args.weights_dir / f"{name}.pt")
    print(f"Saved {args.weights_dir / f'{name}.pt'} (yolov5 commit {commit[:8]})")


if __name__ == "__main__":
    main()
