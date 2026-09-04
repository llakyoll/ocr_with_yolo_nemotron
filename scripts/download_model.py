"""Download and verify the fixed YOLOv11 Large ONNX model."""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPOSITORY = "morsetechlab/yolov11-license-plate-detection"
MODEL_REVISION = "251a30d"
MODEL_FILENAME = "license-plate-finetune-v1l.onnx"
MODEL_SHA256 = "5efdfbe4909bfa6c895bed48676b7de695bf71788932e095e7bc74b8b52b75d8"
MODEL_DIRECTORY = Path("models")


def verify_sha256(path: Path, expected_sha256: str) -> bool:
    """Return whether *path* matches the expected hexadecimal SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256.lower()


def download_model() -> Path:
    """Download the configured model and reject it if its checksum differs."""
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIRECTORY,
        )
    )
    if not verify_sha256(downloaded_path, MODEL_SHA256):
        raise RuntimeError(f"SHA-256 verification failed for {downloaded_path}")
    return downloaded_path


if __name__ == "__main__":
    model_path = download_model()
    print(f"Downloaded and verified model: {model_path}")
