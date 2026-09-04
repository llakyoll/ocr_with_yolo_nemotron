import json
from pathlib import Path

import cv2
import numpy as np

from plate_ocr.config import DetectionConfig
from plate_ocr.detection.types import Detection
from plate_ocr.ocr.nim_client import NimOcrError
from plate_ocr.ocr.types import OcrResult
from plate_ocr.processing.video import run_annotated_video


class FakeDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection((1, 1, 4, 3), 0.9)]


class FakeOcrClient:
    def read_crop(self, crop_path: Path) -> OcrResult:
        return OcrResult("34 AB 123", "34AB123", 0.99, {"data": []})


class FailingOcrClient:
    def read_crop(self, crop_path: Path) -> OcrResult:
        raise NimOcrError("OCR request failed")


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (6, 4))
    assert writer.isOpened()
    writer.write(np.full((4, 6, 3), 10, dtype=np.uint8))
    writer.write(np.full((4, 6, 3), 20, dtype=np.uint8))
    writer.release()


def test_run_annotated_video_writes_results_crops_and_video(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    _write_tiny_video(video_path)
    config = DetectionConfig(video_path, tmp_path / "model.onnx", tmp_path / "runs")

    result = run_annotated_video(config, FakeDetector(), FakeOcrClient())

    assert result.frame_count == 2
    assert (result.detection_count, result.crop_count) == (2, 2)
    assert (result.ocr_success_count, result.ocr_error_count) == (2, 0)
    detection_records = [json.loads(line) for line in (result.run_dir / "detections.jsonl").read_text().splitlines()]
    ocr_records = [json.loads(line) for line in (result.run_dir / "ocr_results.jsonl").read_text().splitlines()]
    assert len(detection_records) == len(ocr_records) == 2
    assert ocr_records[0]["ocr_text"] == "34AB123"
    assert (result.run_dir / "crops" / "frame_000000_det_00.jpg").is_file()
    assert (result.run_dir / "raw_ocr" / "frame_0_det_0.json").is_file()
    capture = cv2.VideoCapture(str(result.run_dir / "annotated.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()


def test_run_annotated_video_records_ocr_errors_and_keeps_writing(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    _write_tiny_video(video_path)
    config = DetectionConfig(video_path, tmp_path / "model.onnx", tmp_path / "runs")

    result = run_annotated_video(config, FakeDetector(), FailingOcrClient())

    records = [json.loads(line) for line in (result.run_dir / "ocr_results.jsonl").read_text().splitlines()]
    assert (result.ocr_success_count, result.ocr_error_count) == (0, 2)
    assert [record["ocr_text"] for record in records] == [None, None]
    assert [record["error"] for record in records] == ["OCR request failed", "OCR request failed"]
    assert (result.run_dir / "annotated.mp4").is_file()
