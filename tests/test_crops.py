import numpy as np

from plate_ocr.detection.types import Detection
from plate_ocr.processing.crops import crop_detection


def test_crop_detection_clamps_box_to_image_bounds() -> None:
    frame = np.full((4, 5, 3), 7, dtype=np.uint8)

    crop = crop_detection(frame, Detection((-2, 1, 7, 4), 0.9))

    assert crop is not None
    assert crop.shape == (3, 5, 3)
    assert np.array_equal(crop, frame[1:4, 0:5])


def test_crop_detection_returns_none_for_zero_area_box() -> None:
    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    assert crop_detection(frame, Detection((2, 1, 2, 3), 0.9)) is None


def test_crop_detection_rounds_float_coordinates_outward() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    crop = crop_detection(frame, Detection((1.2, 2.8, 4.1, 5.01), 0.9))

    assert crop is not None
    assert crop.shape == (4, 4, 3)
