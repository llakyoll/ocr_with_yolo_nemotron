import json
from pathlib import Path

import cv2
import numpy as np

from plate_ocr.ocr.runner import run_ocr
from plate_ocr.ocr.types import OcrResult


class FakeOcrClient:
    def __init__(self) -> None:
        self._results = iter(
            [
                OcrResult("34 AB 123", "34AB123", 0.99, {"request": 1}),
                OcrResult("06 XYZ 99", "06XYZ99", 0.88, {"request": 2}),
            ]
        )

    def read_crop(self, crop_path: Path) -> OcrResult:
        return next(self._results)


def _detection_run_with_two_crops(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True)
    records = []
    for index in range(2):
        crop_path = crops_dir / f"crop-{index}.jpg"
        assert cv2.imwrite(str(crop_path), np.full((4, 6, 3), 127, dtype=np.uint8))
        records.append(
            {
                "detection_id": f"det-{index}",
                "crop_path": f"crops/{crop_path.name}",
                "frame_index": index,
            }
        )
    (run_dir / "detections.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run_dir


def test_run_ocr_writes_raw_response_and_results_for_every_crop(tmp_path: Path) -> None:
    run_dir = _detection_run_with_two_crops(tmp_path)

    result = run_ocr(run_dir, FakeOcrClient())

    records = [
        json.loads(line)
        for line in (result.run_dir / "ocr_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert (result.processed_count, result.success_count, result.error_count) == (2, 2, 0)
    assert [record["ocr_text"] for record in records] == ["34AB123", "06XYZ99"]
    assert all(record["ocr_latency_ms"] >= 0 for record in records)
    assert all(
        (result.run_dir / "raw_ocr" / f"{record['detection_id']}.json").is_file()
        for record in records
    )
