"""Video processing pipeline that writes plate crops and JSONL events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from plate_ocr.config import DetectionConfig
from plate_ocr.detection.types import Detection
from plate_ocr.processing.crops import crop_detection


class Detector(Protocol):
    """The narrow inference interface required by the video pipeline."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame."""


@dataclass(frozen=True)
class RunResult:
    """Locations and counters emitted by one completed pipeline run."""

    run_dir: Path
    detection_count: int
    crop_count: int


def _create_run_dir(runs_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = runs_dir / run_id
    (run_dir / "crops").mkdir(parents=True, exist_ok=False)
    return run_dir


def _record(
    video_id: str,
    frame_index: int,
    timestamp_ms: int,
    detection_index: int,
    detection: Detection,
    crop_path: str | None,
    error: str | None,
) -> dict[str, object]:
    return {
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "detection_id": f"frame_{frame_index}_det_{detection_index}",
        "bbox_xyxy": list(detection.bbox_xyxy),
        "det_confidence": detection.confidence,
        "crop_path": crop_path,
        "error": error,
    }


def run_detection(config: DetectionConfig, detector: Detector) -> RunResult:
    """Process a video and write one JSONL record per model detection."""
    if not config.video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {config.video_path}")
    run_dir = _create_run_dir(config.runs_dir)
    records_path = run_dir / "detections.jsonl"
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {config.video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid FPS: {config.video_path}")

    detection_count = 0
    crop_count = 0
    frame_index = 0
    with records_path.open("w", encoding="utf-8") as records_file:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % config.frame_stride == 0:
                timestamp_ms = round(frame_index * 1000 / fps)
                for detection_index, detection in enumerate(detector.detect(frame)):
                    detection_count += 1
                    crop = crop_detection(frame, detection)
                    if crop is None:
                        record = _record(
                            config.video_path.stem,
                            frame_index,
                            timestamp_ms,
                            detection_index,
                            detection,
                            None,
                            "empty crop",
                        )
                    else:
                        crop_name = f"frame_{frame_index:06d}_det_{detection_index:02d}.jpg"
                        relative_crop_path = Path("crops") / crop_name
                        if not cv2.imwrite(str(run_dir / relative_crop_path), crop):
                            raise OSError(f"could not write crop: {relative_crop_path}")
                        crop_count += 1
                        record = _record(
                            config.video_path.stem,
                            frame_index,
                            timestamp_ms,
                            detection_index,
                            detection,
                            relative_crop_path.as_posix(),
                            None,
                        )
                    records_file.write(json.dumps(record) + "\n")
            frame_index += 1
    capture.release()
    return RunResult(run_dir, detection_count, crop_count)
