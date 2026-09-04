"""Validated runtime configuration for plate detection."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """Inputs and inference settings for one video detection run."""

    video_path: Path
    model_path: Path
    runs_dir: Path
    confidence: float = 0.35
    image_size: int = 1280
    frame_stride: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in the range (0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
