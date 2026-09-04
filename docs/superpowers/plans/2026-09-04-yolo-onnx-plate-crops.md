# YOLOv11 ONNX Plaka Crop Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Örnek videodaki plaka tespitlerini Large YOLOv11 ONNX modeliyle crop ve JSONL kayıtları olarak üretmek.

**Architecture:** `Detector` protokolü ile Ultralytics ONNX bağımlılığı video pipeline'ından ayrılır. Pipeline OpenCV karelerini okur, saf bir bbox kırpma işlevine verir ve dosya sistemi çıktıları ile JSONL olaylarını atomik olmayan fakat kare bazında izlenebilir biçimde oluşturur.

**Tech Stack:** Python 3.11, Ultralytics, ONNX Runtime, OpenCV, NumPy, Hugging Face Hub, pytest.

## Global Constraints

- Model dosyası `models/license-plate-finetune-v1l.onnx` yolunda bulunur ve Git'e eklenmez.
- Model, `morsetechlab/yolov11-license-plate-detection` deposunun `251a30d` revision'ından indirilir.
- SHA-256 değeri `5efdfbe4909bfa6c895bed48676b7de695bf71788932e095e7bc74b8b52b75d8` olmalıdır.
- Varsayılanlar: `confidence=0.35`, `image_size=1280`, `frame_stride=1`.
- Çıktılar `runs/<run_id>/crops/` ve `runs/<run_id>/detections.jsonl` altında oluşturulur.
- OCR, takip, anotasyonlu video ve CSV bu planın kapsamı dışındadır.

---

### Task 1: Paketleme, ayarlar ve doğrulanmış model indirme

**Files:**
- Create: `pyproject.toml`
- Create: `src/plate_ocr/__init__.py`
- Create: `src/plate_ocr/config.py`
- Create: `scripts/download_model.py`
- Create: `tests/test_config.py`
- Create: `tests/test_download_model.py`

**Interfaces:**
- Produces: `DetectionConfig(video_path: Path, model_path: Path, runs_dir: Path, confidence: float, image_size: int, frame_stride: int)` and `verify_sha256(path: Path, expected: str) -> bool`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_config_accepts_default_detection_values(tmp_path):
    config = DetectionConfig(tmp_path / 'in.mp4', tmp_path / 'model.onnx', tmp_path / 'runs')
    assert (config.confidence, config.image_size, config.frame_stride) == (0.35, 1280, 1)

def test_config_rejects_invalid_confidence(tmp_path):
    with pytest.raises(ValueError, match='confidence'):
        DetectionConfig(tmp_path/'in.mp4', tmp_path/'m.onnx', tmp_path/'runs', confidence=1.1)
```

- [ ] **Step 2: Run the configuration test to verify RED**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `plate_ocr.config` does not exist.

- [ ] **Step 3: Implement packaging and `DetectionConfig`**

```python
@dataclass(frozen=True)
class DetectionConfig:
    video_path: Path
    model_path: Path
    runs_dir: Path
    confidence: float = 0.35
    image_size: int = 1280
    frame_stride: int = 1
```

Reject confidence outside `(0, 1]`, non-positive image sizes, and non-positive frame strides.

- [ ] **Step 4: Re-run the configuration test to verify GREEN**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing SHA-256 verification tests**

```python
def test_verify_sha256_accepts_matching_file(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    assert verify_sha256(path, hashlib.sha256(b'model').hexdigest())

def test_verify_sha256_rejects_mismatched_file(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    assert not verify_sha256(path, '0' * 64)
```

- [ ] **Step 6: Run downloader test to verify RED; implement `verify_sha256`; then verify GREEN**

Run: `python -m pytest tests/test_download_model.py -v`

Expected before implementation: FAIL importing `scripts.download_model`; expected after implementation: PASS.

- [ ] **Step 7: Add dependencies and commit Task 1**

```bash
git add pyproject.toml src/plate_ocr tests scripts/download_model.py
git commit -m "feat: add detection configuration and model downloader"
```

### Task 2: Tespit adaptörü ve güvenli crop işlemi

**Files:**
- Create: `src/plate_ocr/detection/__init__.py`
- Create: `src/plate_ocr/detection/types.py`
- Create: `src/plate_ocr/detection/yolo.py`
- Create: `src/plate_ocr/processing/__init__.py`
- Create: `src/plate_ocr/processing/crops.py`
- Create: `tests/test_crops.py`

**Interfaces:**
- Consumes: `DetectionConfig`.
- Produces: `Detection(bbox_xyxy: tuple[float, float, float, float], confidence: float)`; `crop_detection(frame: np.ndarray, detection: Detection) -> np.ndarray | None`; `UltralyticsOnnxDetector.detect(frame: np.ndarray) -> list[Detection]`.

- [ ] **Step 1: Write failing crop tests**

```python
def test_crop_detection_clamps_box_to_image_bounds():
    frame = np.full((4, 5, 3), 7, dtype=np.uint8)
    crop = crop_detection(frame, Detection((-2, 1, 7, 4), 0.9))
    assert crop.shape == (3, 5, 3)

def test_crop_detection_returns_none_for_zero_area_box():
    assert crop_detection(np.zeros((4, 5, 3), dtype=np.uint8), Detection((2, 1, 2, 3), 0.9)) is None
```

- [ ] **Step 2: Run crop tests to verify RED**

Run: `python -m pytest tests/test_crops.py -v`

Expected: FAIL because `plate_ocr.processing.crops` does not exist.

- [ ] **Step 3: Implement minimal crop function**

Convert bbox coordinates with `floor` for starts and `ceil` for ends, clamp them to `[0, width]` and `[0, height]`, and return `None` when `x2 <= x1` or `y2 <= y1`.

- [ ] **Step 4: Re-run crop tests to verify GREEN**

Run: `python -m pytest tests/test_crops.py -v`

Expected: PASS.

- [ ] **Step 5: Implement the ONNX detector adapter**

Load `YOLO(str(model_path), task='detect')`. In `detect`, call `predict(source=frame, conf=confidence, imgsz=image_size, verbose=False)` and convert the first result's `boxes.xyxy` and `boxes.conf` to `Detection` values.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/plate_ocr/detection src/plate_ocr/processing tests/test_crops.py
git commit -m "feat: add ONNX detection and crop primitives"
```

### Task 3: Video pipeline, JSONL raporu ve CLI

**Files:**
- Create: `src/plate_ocr/pipeline.py`
- Create: `src/plate_ocr/cli.py`
- Create: `tests/test_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `DetectionConfig`, `Detection`, `crop_detection`, and any object with `detect(frame) -> list[Detection]`.
- Produces: `RunResult(run_dir: Path, detection_count: int, crop_count: int)` from `run_detection(config, detector)`.

- [ ] **Step 1: Write a failing pipeline integration test using a fake detector**

```python
def test_pipeline_writes_crop_and_jsonl_record_for_detected_plate(tmp_path, tiny_video):
    result = run_detection(config_for(tiny_video, tmp_path), FakeDetector([Detection((1, 1, 4, 3), 0.9)]))
    record = json.loads((result.run_dir / 'detections.jsonl').read_text().strip())
    assert result.crop_count == 1
    assert (result.run_dir / record['crop_path']).is_file()
    assert record['frame_index'] == 0
```

- [ ] **Step 2: Run pipeline test to verify RED**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: FAIL because `plate_ocr.pipeline` does not exist.

- [ ] **Step 3: Implement the minimal pipeline**

Create a UTC run id, open the video with `cv2.VideoCapture`, skip frames not divisible by `frame_stride`, write crop files using `cv2.imwrite`, and append one JSON object per successful crop. Compute `timestamp_ms` as `round(frame_index * 1000 / fps)`.

- [ ] **Step 4: Re-run pipeline test to verify GREEN**

Run: `python -m pytest tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Add `plate-ocr-detect` CLI and README command**

```bash
plate-ocr-detect --video data/input/sample_plate_video.mp4 --model models/license-plate-finetune-v1l.onnx
```

The command prints run directory, detection count, and crop count; it exits non-zero for missing video/model or unreadable video.

- [ ] **Step 6: Run all automated tests and commit Task 3**

Run: `python -m pytest -v`

Expected: PASS.

```bash
git add src/plate_ocr tests README.md
git commit -m "feat: add video plate crop pipeline"
```

### Task 4: Gerçek model ve örnek video doğrulaması

**Files:**
- Modify: ignored `models/license-plate-finetune-v1l.onnx`
- Create: ignored `runs/<run_id>/`

**Interfaces:**
- Consumes: Task 1-3 CLI and downloaded model.
- Produces: Example-video crop output and JSONL report.

- [ ] **Step 1: Download and verify the ONNX model**

Run: `python scripts/download_model.py`

Expected: model path and matching SHA-256 are printed.

- [ ] **Step 2: Process the complete sample video**

Run: `plate-ocr-detect --video data/input/sample_plate_video.mp4 --model models/license-plate-finetune-v1l.onnx`

Expected: a new run directory, at least one crop, and `detections.jsonl` are printed.

- [ ] **Step 3: Verify output integrity**

Run: `find runs -path '*/crops/*.jpg' -type f | wc -l` and `python -c "import json; [json.loads(x) for x in open('runs/<run_id>/detections.jsonl')]; print('valid jsonl')"`

Expected: crop count is positive and JSONL parses without errors.
