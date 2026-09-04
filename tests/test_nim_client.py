import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from plate_ocr.ocr.nim_client import NimOcrClient, NimOcrError


def _write_jpeg(path: Path) -> Path:
    assert cv2.imwrite(str(path), np.full((4, 6, 3), 127, dtype=np.uint8))
    return path


def test_read_crop_posts_jpeg_data_url_and_parses_word_detections(tmp_path: Path) -> None:
    observed_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "text_detections": [
                            {"text_prediction": {"text": "34 AB", "confidence": 0.97}},
                            {"text_prediction": {"text": "123", "confidence": 0.99}},
                        ]
                    }
                ]
            },
        )

    client = NimOcrClient("http://nim.test", transport=httpx.MockTransport(handler))

    result = client.read_crop(_write_jpeg(tmp_path / "crop.jpg"))

    assert result.raw_text == "34 AB 123"
    assert result.text == "34AB123"
    assert result.confidence == 0.98
    assert observed_payload["merge_levels"] == ["word"]
    image = observed_payload["input"][0]
    assert image["type"] == "image_url"
    assert image["url"].startswith("data:image/jpeg;base64,")


def test_read_crop_raises_nim_error_for_http_failure(tmp_path: Path) -> None:
    client = NimOcrClient(
        "http://nim.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="unavailable")),
    )

    with pytest.raises(NimOcrError, match="503"):
        client.read_crop(_write_jpeg(tmp_path / "crop.jpg"))
