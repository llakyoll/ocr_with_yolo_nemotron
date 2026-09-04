from pathlib import Path

import pytest

from plate_ocr.config import DetectionConfig


def test_config_accepts_default_detection_values(tmp_path: Path) -> None:
    config = DetectionConfig(
        video_path=tmp_path / "input.mp4",
        model_path=tmp_path / "model.onnx",
        runs_dir=tmp_path / "runs",
    )

    assert (config.confidence, config.image_size, config.frame_stride) == (0.35, 1280, 1)


@pytest.mark.parametrize(
    ("confidence", "image_size", "frame_stride", "message"),
    [
        (0.0, 1280, 1, "confidence"),
        (1.01, 1280, 1, "confidence"),
        (0.35, 0, 1, "image_size"),
        (0.35, 1280, 0, "frame_stride"),
    ],
)
def test_config_rejects_invalid_detection_values(
    tmp_path: Path,
    confidence: float,
    image_size: int,
    frame_stride: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DetectionConfig(
            video_path=tmp_path / "input.mp4",
            model_path=tmp_path / "model.onnx",
            runs_dir=tmp_path / "runs",
            confidence=confidence,
            image_size=image_size,
            frame_stride=frame_stride,
        )
