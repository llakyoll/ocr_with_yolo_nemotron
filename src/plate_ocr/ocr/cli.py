"""Command-line entry point for OCR replay over an existing detection run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plate_ocr.ocr.nim_client import NimOcrClient
from plate_ocr.ocr.runner import run_ocr


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse OCR replay arguments without contacting the NIM service."""
    parser = argparse.ArgumentParser(description="Read detected plate crops with Nemotron OCR.")
    parser.add_argument("--run", type=Path, required=True, help="Detection run directory")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="Nemotron OCR NIM base URL"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-crop OCR timeout in seconds")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run OCR replay and print processed, successful, and failed counters."""
    parsed = parse_arguments(arguments)
    if not parsed.run.is_dir():
        raise SystemExit(f"run directory does not exist: {parsed.run}")
    client = NimOcrClient(parsed.base_url, timeout_seconds=parsed.timeout)
    try:
        result = run_ocr(parsed.run, client)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    print(f"Run directory: {result.run_dir}")
    print(f"Processed: {result.processed_count}")
    print(f"Succeeded: {result.success_count}")
    print(f"Failed: {result.error_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
