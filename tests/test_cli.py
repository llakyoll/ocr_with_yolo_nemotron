from pathlib import Path

from plate_ocr.cli import parse_arguments


def test_parse_arguments_uses_detection_defaults() -> None:
    arguments = parse_arguments(["--video", "input.mp4", "--model", "model.onnx"])

    assert arguments.video == Path("input.mp4")
    assert arguments.model == Path("model.onnx")
    assert arguments.runs_dir == Path("runs")
    assert (arguments.confidence, arguments.image_size, arguments.frame_stride) == (0.35, 1280, 1)
