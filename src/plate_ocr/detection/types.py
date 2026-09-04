"""Types shared by detection and crop processing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected plate in pixel coordinates, ordered x1, y1, x2, y2."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
