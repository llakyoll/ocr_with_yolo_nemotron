import json
from pathlib import Path

import cv2
import numpy as np

from plate_ocr.config import DetectionConfig
from plate_ocr.detection.types import Detection
from plate_ocr.pipeline import run_detection


class FakeDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection((1, 1, 4, 3), 0.9)]


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (6, 4))
    assert writer.isOpened()
    writer.write(np.full((4, 6, 3), 10, dtype=np.uint8))
    writer.write(np.full((4, 6, 3), 20, dtype=np.uint8))
    writer.release()


def test_pipeline_writes_crop_and_jsonl_record_for_each_detected_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    _write_tiny_video(video_path)
    config = DetectionConfig(video_path, tmp_path / "model.onnx", tmp_path / "runs")

    result = run_detection(config, FakeDetector())

    records = [
        json.loads(line)
        for line in (result.run_dir / "detections.jsonl").read_text().splitlines()
    ]
    assert result.detection_count == 2
    assert result.crop_count == 2
    assert [record["frame_index"] for record in records] == [0, 1]
    assert [record["timestamp_ms"] for record in records] == [0, 100]
    assert all((result.run_dir / record["crop_path"]).is_file() for record in records)
    assert all(record["bbox_xyxy"] == [1.0, 1.0, 4.0, 3.0] for record in records)


def test_pipeline_skips_empty_crops_but_counts_detection(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    _write_tiny_video(video_path)
    config = DetectionConfig(video_path, tmp_path / "model.onnx", tmp_path / "runs")

    result = run_detection(config, FakeDetectorWithEmptyBox())

    assert result.detection_count == 2
    assert result.crop_count == 0
    records = [
        json.loads(line)
        for line in (result.run_dir / "detections.jsonl").read_text().splitlines()
    ]
    assert [record["error"] for record in records] == ["empty crop", "empty crop"]


class FakeDetectorWithEmptyBox:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection((2, 1, 2, 3), 0.9)]
