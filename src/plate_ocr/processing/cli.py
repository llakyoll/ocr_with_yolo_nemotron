"""CLI for a fresh detection, OCR, and annotated-video run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plate_ocr.config import DetectionConfig
from plate_ocr.detection.yolo import UltralyticsOnnxDetector
from plate_ocr.ocr.nim_client import NimOcrClient
from plate_ocr.processing.video import run_annotated_video


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect plates, read them, and write an annotated video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    if not parsed.video.is_file() or not parsed.model.is_file():
        raise SystemExit("video and model files must exist")
    try:
        config = DetectionConfig(parsed.video, parsed.model, parsed.runs_dir, parsed.confidence, parsed.image_size, parsed.frame_stride)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = run_annotated_video(
        config,
        UltralyticsOnnxDetector(config.model_path, config.confidence, config.image_size),
        NimOcrClient(parsed.base_url, timeout_seconds=parsed.timeout),
    )
    print(f"Run directory: {result.run_dir}")
    print(f"Frames: {result.frame_count}; detections: {result.detection_count}; OCR errors: {result.ocr_error_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
