import hashlib
from pathlib import Path

from scripts.download_model import verify_sha256


def test_verify_sha256_accepts_matching_file(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")

    assert verify_sha256(path, hashlib.sha256(b"model").hexdigest())


def test_verify_sha256_rejects_mismatched_file(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")

    assert not verify_sha256(path, "0" * 64)
