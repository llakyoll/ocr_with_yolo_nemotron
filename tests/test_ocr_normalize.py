from plate_ocr.ocr.normalize import normalize_plate_text


def test_normalize_plate_text_preserves_raw_meaning_without_substitution() -> None:
    assert normalize_plate_text(" 34 ab-123 ") == "34AB123"


def test_normalize_plate_text_does_not_substitute_ambiguous_characters() -> None:
    assert normalize_plate_text("O1 S5") == "O1S5"


def test_normalize_plate_text_keeps_turkish_alphanumeric_characters() -> None:
    assert normalize_plate_text(" ç 34-ü ") == "Ç34Ü"
