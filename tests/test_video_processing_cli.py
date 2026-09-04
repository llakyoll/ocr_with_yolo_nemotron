from pathlib import Path
import subprocess
import sys

from plate_ocr.processing.cli import parse_arguments


def test_parse_arguments_uses_detection_and_nim_defaults() -> None:
    arguments = parse_arguments(["--video", "input.mp4", "--model", "model.onnx"])

    assert arguments.runs_dir == Path("runs")
    assert arguments.base_url == "http://127.0.0.1:8000"
    assert (arguments.confidence, arguments.image_size, arguments.frame_stride) == (0.35, 1280, 1)
    assert arguments.timeout == 30.0


def test_module_entry_point_displays_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "plate_ocr.processing.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Detect plates" in completed.stdout
