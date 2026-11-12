"""Common interface every detector in the benchmark implements.

A detector takes one BGR image (as returned by cv2.imread) and returns all of its detections
for that image. Everything a deployed model would do per frame (resizing, normalisation,
inference, NMS, rescaling boxes to the original image) happens inside `predict`, so the
speed benchmark measures the same thing for every model.

For mAP, return low-confidence detections too (e.g. score >= 0.001): COCO AP integrates over
the whole precision-recall curve, and a high threshold under-reports it.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detections:
    """Detections for one image.

    boxes:  (N, 4) float array, [x1, y1, x2, y2] in pixels of the original image
    scores: (N,) float array in [0, 1]
    labels: (N,) int array, 0-based class index into the detector's class_names
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if not len(self.boxes) == len(self.scores) == len(self.labels):
            raise ValueError("boxes, scores and labels must have the same length")

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0))


class Detector(Protocol):
    name: str
    class_names: list[str]

    def predict(self, image: np.ndarray) -> Detections: ...
