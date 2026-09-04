import numpy as np

from plate_ocr.detection.yolo import detections_from_result


class FakeBoxes:
    xyxy = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    conf = np.array([0.91, 0.42])


class FakeResult:
    boxes = FakeBoxes()


def test_detections_from_result_preserves_boxes_and_confidences() -> None:
    detections = detections_from_result(FakeResult())

    assert [detection.bbox_xyxy for detection in detections] == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ]
    assert [detection.confidence for detection in detections] == [0.91, 0.42]
