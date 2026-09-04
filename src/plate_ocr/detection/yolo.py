"""Ultralytics ONNX adapter for license-plate detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from plate_ocr.detection.types import Detection


def _to_list(value: Any) -> list[Any]:
    """Convert NumPy arrays and Torch tensors returned by Ultralytics to lists."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def detections_from_result(result: Any) -> list[Detection]:
    """Map the first Ultralytics result's boxes to dependency-free records."""
    if result.boxes is None:
        return []
    boxes = _to_list(result.boxes.xyxy)
    confidences = _to_list(result.boxes.conf)
    return [
        Detection(tuple(map(float, box)), float(confidence))
        for box, confidence in zip(boxes, confidences, strict=True)
    ]


class UltralyticsOnnxDetector:
    """Run a fixed ONNX YOLO model against BGR OpenCV frames."""

    def __init__(self, model_path: Path, confidence: float, image_size: int) -> None:
        self._model = YOLO(str(model_path), task="detect")
        self._confidence = confidence
        self._image_size = image_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return all model detections for a single video frame."""
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._image_size,
            verbose=False,
        )
        return detections_from_result(results[0])
