"""Single-pass detection, OCR, and annotated-video processing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from plate_ocr.config import DetectionConfig
from plate_ocr.ocr.nim_client import NimOcrError
from plate_ocr.ocr.runner import CropOcrClient
from plate_ocr.pipeline import Detector, _create_run_dir, _record
from plate_ocr.processing.annotation import FrameAnnotation, annotate_frame
from plate_ocr.processing.crops import crop_detection


@dataclass(frozen=True)
class AnnotatedVideoRunResult:
    """Locations and counters emitted by one annotated video run."""

    run_dir: Path
    frame_count: int
    detection_count: int
    crop_count: int
    ocr_success_count: int
    ocr_error_count: int


def run_annotated_video(
    config: DetectionConfig, detector: Detector, client: CropOcrClient
) -> AnnotatedVideoRunResult:
    """Write detection/OCR records, crops, raw responses, and a silent MP4."""
    if not config.video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {config.video_path}")
    run_dir = _create_run_dir(config.runs_dir)
    raw_dir = run_dir / "raw_ocr"
    raw_dir.mkdir()
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {config.video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid metadata: {config.video_path}")
    video_path = run_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open annotated video writer: {video_path}")

    frame_count = detection_count = crop_count = ocr_success_count = ocr_error_count = 0
    try:
        with (run_dir / "detections.jsonl").open("w", encoding="utf-8") as detections_file, (
            run_dir / "ocr_results.jsonl"
        ).open("w", encoding="utf-8") as ocr_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                annotations: list[FrameAnnotation] = []
                if frame_count % config.frame_stride == 0:
                    timestamp_ms = round(frame_count * 1000 / fps)
                    for detection_index, detection in enumerate(detector.detect(frame)):
                        detection_count += 1
                        crop = crop_detection(frame, detection)
                        crop_path: str | None = None
                        error: str | None = None
                        ocr_raw_text: str | None = None
                        ocr_text: str | None = None
                        ocr_confidence: float | None = None
                        ocr_latency_ms: float | None = None
                        if crop is None:
                            error = "empty crop"
                        else:
                            crop_name = f"frame_{frame_count:06d}_det_{detection_index:02d}.jpg"
                            crop_path = (Path("crops") / crop_name).as_posix()
                            if not cv2.imwrite(str(run_dir / crop_path), crop):
                                raise OSError(f"could not write crop: {crop_path}")
                            crop_count += 1
                            started_at = perf_counter()
                            try:
                                ocr_result = client.read_crop(run_dir / crop_path)
                            except NimOcrError as exc:
                                ocr_error_count += 1
                                error = str(exc)
                            else:
                                ocr_success_count += 1
                                ocr_raw_text = ocr_result.raw_text
                                ocr_text = ocr_result.text
                                ocr_confidence = ocr_result.confidence
                                (raw_dir / f"frame_{frame_count}_det_{detection_index}.json").write_text(
                                    json.dumps(ocr_result.raw_response, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                            ocr_latency_ms = round((perf_counter() - started_at) * 1000, 3)
                        detection_record = _record(
                            config.video_path.stem,
                            frame_count,
                            timestamp_ms,
                            detection_index,
                            detection,
                            crop_path,
                            error,
                        )
                        detections_file.write(json.dumps(detection_record, ensure_ascii=False) + "\n")
                        ocr_record = dict(detection_record)
                        ocr_record.update(
                            {
                                "ocr_raw_text": ocr_raw_text,
                                "ocr_text": ocr_text,
                                "ocr_confidence": ocr_confidence,
                                "ocr_latency_ms": ocr_latency_ms,
                            }
                        )
                        ocr_file.write(json.dumps(ocr_record, ensure_ascii=False) + "\n")
                        annotations.append(FrameAnnotation(detection, ocr_text))
                writer.write(annotate_frame(frame, annotations))
                frame_count += 1
    finally:
        writer.release()
        capture.release()
    return AnnotatedVideoRunResult(
        run_dir, frame_count, detection_count, crop_count, ocr_success_count, ocr_error_count
    )
