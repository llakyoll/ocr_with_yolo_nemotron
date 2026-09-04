"""Structured OCR result values independent of the NIM response schema."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrResult:
    """Text extracted from one crop and the unmodified NIM response."""

    raw_text: str
    text: str
    confidence: float | None
    raw_response: dict[str, object]
