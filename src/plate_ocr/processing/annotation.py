"""Draw plate detection and OCR labels onto BGR video frames."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from plate_ocr.detection.types import Detection


@dataclass(frozen=True)
class FrameAnnotation:
    """A detected plate and the OCR text to render for it."""

    detection: Detection
    ocr_text: str | None


def label_for_ocr_text(ocr_text: str | None) -> str:
    """Return a viewer-facing label even when OCR did not yield text."""
    return ocr_text if ocr_text else "OKUNAMADI"


def annotate_frame(frame: np.ndarray, annotations: list[FrameAnnotation]) -> np.ndarray:
    """Return a copied frame with bounds-clamped green boxes and OCR labels."""
    output = frame.copy()
    height, width = output.shape[:2]
    for annotation in annotations:
        x1, y1, x2, y2 = annotation.detection.bbox_xyxy
        left = max(0, min(width - 1, floor(x1)))
        top = max(0, min(height - 1, floor(y1)))
        right = max(0, min(width - 1, ceil(x2)))
        bottom = max(0, min(height - 1, ceil(y2)))
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        label = label_for_ocr_text(annotation.ocr_text)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        label_top = top - text_height - baseline - 6
        if label_top < 0:
            label_top = top
        label_bottom = min(height - 1, label_top + text_height + baseline + 6)
        label_right = min(width - 1, left + text_width + 8)
        cv2.rectangle(output, (left, label_top), (label_right, label_bottom), (0, 96, 0), -1)
        text_y = min(height - 1, label_top + text_height + 3)
        cv2.putText(
            output,
            label,
            (left + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output
