"""Conservative normalization for OCR plate text."""


def normalize_plate_text(value: str) -> str:
    """Uppercase and retain alphanumeric characters without substitutions."""
    return "".join(character for character in value.upper() if character.isalnum())
