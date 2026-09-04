"""Replay a detection run's crops through an OCR client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from plate_ocr.ocr.nim_client import NimOcrError
from plate_ocr.ocr.types import OcrResult


class CropOcrClient(Protocol):
    """The narrow OCR interface needed to replay a detection run."""

    def read_crop(self, crop_path: Path) -> OcrResult:
        """Return OCR data for a crop file."""


@dataclass(frozen=True)
class OcrRunResult:
    """Counters for a completed OCR replay."""

    run_dir: Path
    processed_count: int
    success_count: int
    error_count: int


def run_ocr(run_dir: Path, client: CropOcrClient) -> OcrRunResult:
    """Write OCR results without modifying the source detection JSONL file."""
    detections_path = run_dir / "detections.jsonl"
    if not detections_path.is_file():
        raise FileNotFoundError(f"detection results do not exist: {detections_path}")
    raw_dir = run_dir / "raw_ocr"
    raw_dir.mkdir(exist_ok=True)
    processed_count = 0
    success_count = 0
    error_count = 0
    with detections_path.open(encoding="utf-8") as detection_file, (run_dir / "ocr_results.jsonl").open(
        "w", encoding="utf-8"
    ) as result_file:
        for line in detection_file:
            detection = json.loads(line)
            crop_relative_path = detection.get("crop_path")
            if not isinstance(crop_relative_path, str):
                continue
            processed_count += 1
            record = dict(detection)
            started_at = perf_counter()
            try:
                result = client.read_crop(run_dir / crop_relative_path)
            except NimOcrError as error:
                error_count += 1
                record.update(
                    {
                        "ocr_raw_text": None,
                        "ocr_text": None,
                        "ocr_confidence": None,
                        "ocr_latency_ms": round((perf_counter() - started_at) * 1000, 3),
                        "error": str(error),
                    }
                )
            else:
                success_count += 1
                detection_id = str(detection["detection_id"])
                (raw_dir / f"{detection_id}.json").write_text(
                    json.dumps(result.raw_response, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                record.update(
                    {
                        "ocr_raw_text": result.raw_text,
                        "ocr_text": result.text,
                        "ocr_confidence": result.confidence,
                        "ocr_latency_ms": round((perf_counter() - started_at) * 1000, 3),
                        "error": None,
                    }
                )
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return OcrRunResult(run_dir, processed_count, success_count, error_count)
