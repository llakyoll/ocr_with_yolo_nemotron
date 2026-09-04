"""Safe conversion of detections into image crops."""

from __future__ import annotations

from math import ceil, floor

import numpy as np

from plate_ocr.detection.types import Detection


def crop_detection(frame: np.ndarray, detection: Detection) -> np.ndarray | None:
    """Return a bounds-clamped crop, or ``None`` for an empty bounding box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    left = max(0, min(width, floor(x1)))
    top = max(0, min(height, floor(y1)))
    right = max(0, min(width, ceil(x2)))
    bottom = max(0, min(height, ceil(y2)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]
