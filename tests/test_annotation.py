import numpy as np

from plate_ocr.detection.types import Detection
from plate_ocr.processing.annotation import FrameAnnotation, annotate_frame, label_for_ocr_text


def test_annotate_frame_draws_a_clamped_green_plate_box() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)

    annotated = annotate_frame(
        frame,
        [FrameAnnotation(Detection((-4, 6, 30, 22), 0.9), "34AB123")],
    )

    assert annotated is not frame
    assert np.array_equal(frame, np.zeros((40, 80, 3), dtype=np.uint8))
    assert annotated[6, 0, 1] > 0


def test_label_for_ocr_text_marks_missing_text_as_unreadable() -> None:
    assert label_for_ocr_text(None) == "OKUNAMADI"
    assert label_for_ocr_text("") == "OKUNAMADI"
    assert label_for_ocr_text("34AB123") == "34AB123"
