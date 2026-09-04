from pathlib import Path

from plate_ocr.ocr.cli import parse_arguments


def test_parse_arguments_uses_local_nim_default_url() -> None:
    arguments = parse_arguments(["--run", "runs/example"])

    assert arguments.run == Path("runs/example")
    assert arguments.base_url == "http://127.0.0.1:8000"
