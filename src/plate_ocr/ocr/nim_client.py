"""HTTP adapter for NVIDIA Nemotron OCR v2 NIM."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from plate_ocr.ocr.normalize import normalize_plate_text
from plate_ocr.ocr.types import OcrResult


class NimOcrError(RuntimeError):
    """An OCR request failed or produced an invalid NIM response."""


class NimOcrClient:
    """Read individual JPEG crop files through the NIM `/v1/ocr` API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def read_crop(self, crop_path: Path) -> OcrResult:
        """Submit one JPEG crop and convert the NIM word detections to text."""
        try:
            encoded_image = base64.b64encode(crop_path.read_bytes()).decode("ascii")
        except OSError as error:
            raise NimOcrError(f"could not read crop {crop_path}: {error}") from error
        payload = {
            "input": [{"type": "image_url", "url": f"data:image/jpeg;base64,{encoded_image}"}],
            "merge_levels": ["word"],
        }
        try:
            response = self._client.post(f"{self._base_url}/v1/ocr", json=payload)
        except httpx.HTTPError as error:
            raise NimOcrError(f"OCR request failed: {error}") from error
        if response.is_error:
            raise NimOcrError(f"OCR request returned {response.status_code}: {response.text}")
        try:
            raw_response = response.json()
            detections = raw_response["data"][0]["text_detections"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise NimOcrError("OCR response does not match the expected NIM schema") from error
        texts: list[str] = []
        confidences: list[float] = []
        for detection in detections:
            prediction = detection.get("text_prediction", {})
            text = prediction.get("text")
            confidence = prediction.get("confidence")
            if isinstance(text, str) and text:
                texts.append(text)
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))
        raw_text = " ".join(texts)
        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        return OcrResult(raw_text, normalize_plate_text(raw_text), mean_confidence, raw_response)
