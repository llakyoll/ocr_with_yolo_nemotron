"""Command-line entry point for video plate detection and crop creation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plate_ocr.config import DetectionConfig
from plate_ocr.detection.yolo import UltralyticsOnnxDetector
from plate_ocr.pipeline import run_detection


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without running inference."""
    parser = argparse.ArgumentParser(description="Detect license plates and save crops.")
    parser.add_argument("--video", type=Path, required=True, help="Input MP4/MOV/AVI/MKV video")
    parser.add_argument("--model", type=Path, required=True, help="YOLO ONNX model path")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Output root directory")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--image-size", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run detection and print the resulting run location and counters."""
    parsed = parse_arguments(arguments)
    if not parsed.video.is_file():
        raise SystemExit(f"video file does not exist: {parsed.video}")
    if not parsed.model.is_file():
        raise SystemExit(f"model file does not exist: {parsed.model}")
    try:
        config = DetectionConfig(
            video_path=parsed.video,
            model_path=parsed.model,
            runs_dir=parsed.runs_dir,
            confidence=parsed.confidence,
            image_size=parsed.image_size,
            frame_stride=parsed.frame_stride,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    detector = UltralyticsOnnxDetector(config.model_path, config.confidence, config.image_size)
    result = run_detection(config, detector)
    print(f"Run directory: {result.run_dir}")
    print(f"Detections: {result.detection_count}")
    print(f"Crops: {result.crop_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
