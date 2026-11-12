"""YOLO detectors behind the common Detector interface.

UltralyticsDetector covers YOLOv8 / YOLO11 / YOLO26 (the `ultralytics` package).
YOLOv5Detector loads weights trained with the original ultralytics/yolov5 repo.
"""

from pathlib import Path

import numpy as np

from tdb.models.base import Detections


def _check_names(model_names: dict | list, class_names: list[str]) -> None:
    names = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
    if names != list(class_names):
        raise ValueError(
            f"Weights were trained on classes {names}, but the dataset has {class_names}. "
            "Evaluating on the wrong dataset variant?"
        )


class UltralyticsDetector:
    def __init__(
        self,
        weights: str,
        class_names: list[str],
        conf: float = 0.001,
        iou: float = 0.6,
        max_det: int = 300,
        imgsz: int = 640,
        device: str | int | None = None,
        half: bool = False,
    ):
        from ultralytics import YOLO

        self.weights = weights
        self.name = Path(weights).stem
        self.class_names = class_names
        self.model = YOLO(weights)
        _check_names(self.model.names, class_names)
        self.kwargs = dict(
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if half:  # passing half=False still triggers a deprecation warning in Ultralytics 8.4
            self.kwargs["half"] = True
        self.imgsz = imgsz

    @property
    def module(self):
        return self.model.model

    def example_input(self):
        import torch

        device = next(self.module.parameters()).device
        return torch.zeros(1, 3, self.imgsz, self.imgsz, device=device)

    def predict(self, image: np.ndarray) -> Detections:
        boxes = self.model.predict(image, **self.kwargs)[0].boxes  # BGR numpy input
        return Detections(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        )


class YOLOv5Detector:
    """Weights from the original yolov5 repo, loaded through torch.hub (AutoShape)."""

    def __init__(
        self,
        weights: str,
        class_names: list[str],
        repo: str = "yolov5",
        conf: float = 0.001,
        iou: float = 0.6,
        max_det: int = 300,
        imgsz: int = 640,
        device: str | None = None,
    ):
        import torch

        self.weights = weights
        self.name = Path(weights).stem
        self.class_names = class_names
        self.imgsz = imgsz
        source = "local" if Path(repo).is_dir() else "github"
        repo = repo if source == "local" else "ultralytics/yolov5"
        self.model = torch.hub.load(
            repo, "custom", path=weights, source=source, device=device, verbose=False
        )
        self.model.conf, self.model.iou, self.model.max_det = conf, iou, max_det
        _check_names(self.model.names, class_names)

    @property
    def module(self):
        return self.model.model.model  # AutoShape -> DetectMultiBackend -> DetectionModel

    def example_input(self):
        import torch

        device = next(self.module.parameters()).device
        return torch.zeros(1, 3, self.imgsz, self.imgsz, device=device)

    def predict(self, image: np.ndarray) -> Detections:
        out = self.model(image[:, :, ::-1], size=self.imgsz).xyxy[0].cpu().numpy()  # BGR->RGB
        return Detections(out[:, :4], out[:, 4], out[:, 5])
