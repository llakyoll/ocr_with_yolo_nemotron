# Uçtan Uca Anotasyonlu Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kaynak videoyu YOLO ve Nemotron OCR ile baştan işleyerek sonuçları ve sessiz, anotasyonlu MP4 çıktısını tek run dizininde üretmek.

**Architecture:** Yeni anotasyon modülü, bir BGR kare üzerine sınırlandırılmış plaka kutuları ve OCR etiketleri çizer. Yeni uçtan uca pipeline; videoyu bir kez okuyup detector, crop yazımı, OCR istemcisi, JSONL kayıtları ve `cv2.VideoWriter` üzerinden anotasyonlu video üretimini koordine eder. CLI mevcut ONNX detector ve NIM HTTP adaptörlerini oluşturur; ayrı detection ve OCR replay komutları değişmez.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, Ultralytics ONNX Runtime GPU, HTTPX, NVIDIA Nemotron OCR v2 NIM, pytest.

## Global Constraints

- Yeni komut adı tam olarak `plate-ocr-process-video` olmalıdır.
- Her çalışma yeni UTC zaman damgalı `runs/<run_id>/` dizini oluşturmalıdır.
- `detections.jsonl` mevcut detection şemasını korumalı; `ocr_results.jsonl` mevcut OCR replay alanlarını korumalıdır.
- `annotated.mp4` sessiz olmalı, kaynak çözünürlüğünü ve FPS değerini korumalıdır.
- Başarısız OCR yalnızca ilgili kayıtta `error` ve `OKUNAMADI` etiketi üretmeli; video akışı sürmelidir.
- Kaynak video, model, NIM anahtarı, `models/`, `runs/` ve `.nim-cache/` Git'e eklenmemelidir.

---

### Task 1: Kare Anotasyon Modülü

**Files:**
- Create: `src/plate_ocr/processing/annotation.py`
- Test: `tests/test_annotation.py`

**Interfaces:**
- Consumes: `numpy.ndarray` BGR frame, `plate_ocr.detection.types.Detection`, `str | None` OCR text.
- Produces: `annotate_frame(frame: np.ndarray, annotations: list[FrameAnnotation]) -> np.ndarray` ve `FrameAnnotation(detection: Detection, ocr_text: str | None)`.

- [ ] **Step 1: Metin geri dönüşü ve kare sınırlarını kapsayan başarısız testleri yazın**

```python
def test_annotate_frame_draws_green_box_and_normalized_label() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    output = annotate_frame(
        frame,
        [FrameAnnotation(Detection((-4, 6, 30, 22), 0.9), "34AB123")],
    )
    assert output is not frame
    assert np.any(output[:, :, 1] > 0)


def test_annotate_frame_uses_unreadable_label_for_empty_text() -> None:
    assert label_for_ocr_text(None) == "OKUNAMADI"
    assert label_for_ocr_text("") == "OKUNAMADI"
```

- [ ] **Step 2: Testi başarısız olduğunu doğrulayarak çalıştırın**

Run: `pytest tests/test_annotation.py -v`

Expected: `ModuleNotFoundError: No module named 'plate_ocr.processing.annotation'`.

- [ ] **Step 3: Sınırlandırılmış çizim ve etiket fonksiyonunu minimum kapsamla yazın**

```python
@dataclass(frozen=True)
class FrameAnnotation:
    detection: Detection
    ocr_text: str | None


def label_for_ocr_text(ocr_text: str | None) -> str:
    return ocr_text if ocr_text else "OKUNAMADI"


def annotate_frame(frame: np.ndarray, annotations: list[FrameAnnotation]) -> np.ndarray:
    output = frame.copy()
    # floor/ceil koordinatlarını frame sınırlarına kırpın; yeşil kutuyu,
    # arka plan dikdörtgenini ve etiketi cv2.rectangle/cv2.putText ile çizin.
    return output
```

Etiketi kutunun üstüne yerleştirin; üst sınır yetersizse kutunun içine alın.
Boş ya da ters koordinatlı kutuyu çizmeden atlayın.

- [ ] **Step 4: Anotasyon testlerini çalıştırıp başarılı olduklarını doğrulayın**

Run: `pytest tests/test_annotation.py -v`

Expected: PASS.

- [ ] **Step 5: Tamamlanan anotasyon birimini commit edin**

```bash
git add src/plate_ocr/processing/annotation.py tests/test_annotation.py
git commit -m "feat: add plate video annotations"
```

### Task 2: Uçtan Uca Video İşleme Pipeline'ı

**Files:**
- Create: `src/plate_ocr/processing/video.py`
- Modify: `src/plate_ocr/processing/__init__.py`
- Test: `tests/test_video_processing.py`

**Interfaces:**
- Consumes: `DetectionConfig`, `plate_ocr.pipeline.Detector`, `plate_ocr.ocr.runner.CropOcrClient`.
- Produces: `run_annotated_video(config, detector, client) -> AnnotatedVideoRunResult`, burada sonuç `run_dir`, `frame_count`, `detection_count`, `crop_count`, `ocr_success_count`, `ocr_error_count` alanlarını taşır.

- [ ] **Step 1: Küçük yapay video üzerinden başarısız entegrasyon testi yazın**

```python
def test_run_annotated_video_writes_traceable_results_and_silent_mp4(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    _write_tiny_video(video_path)
    config = DetectionConfig(video_path, tmp_path / "model.onnx", tmp_path / "runs")

    result = run_annotated_video(config, FakeDetector(), FakeOcrClient())

    assert result.frame_count == 2
    assert (result.detection_count, result.crop_count) == (2, 2)
    assert (result.ocr_success_count, result.ocr_error_count) == (2, 0)
    assert len((result.run_dir / "detections.jsonl").read_text().splitlines()) == 2
    assert len((result.run_dir / "ocr_results.jsonl").read_text().splitlines()) == 2
    assert (result.run_dir / "raw_ocr" / "frame_0_det_0.json").is_file()
    capture = cv2.VideoCapture(str(result.run_dir / "annotated.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    assert capture.get(cv2.CAP_PROP_AUDIO_TOTAL_STREAMS) == 0
```

`FakeOcrClient` ikinci çağrıda `NimOcrError("OCR request failed")` üretsin;
ek test bu durumda iki video karesi, iki OCR JSONL kaydı ve ikinci kayıt için
`ocr_text is None`, `error` dolu olduğunu doğrulasın.

- [ ] **Step 2: Testi başarısız olduğunu doğrulayarak çalıştırın**

Run: `pytest tests/test_video_processing.py -v`

Expected: `ModuleNotFoundError: No module named 'plate_ocr.processing.video'`.

- [ ] **Step 3: Tek geçişli yazma akışını minimum kapsamla uygulayın**

```python
def run_annotated_video(
    config: DetectionConfig,
    detector: Detector,
    client: CropOcrClient,
) -> AnnotatedVideoRunResult:
    run_dir = _create_video_run_dir(config.runs_dir)
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {config.video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"video has invalid metadata: {config.video_path}")
    writer = cv2.VideoWriter(
        str(run_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open annotated video writer: {run_dir / 'annotated.mp4'}")
    try:
        return _write_processed_frames(capture, writer, run_dir, config, detector, client, fps)
    finally:
        writer.release()
        capture.release()
```

`_write_processed_frames`, `detections.jsonl` ve `ocr_results.jsonl` dosyalarını
birlikte açmalıdır. Her kare için `frame_index % config.frame_stride == 0`
durumunda detector çağrılır; her detection için `crop_detection`, mevcut
`pipeline._record` şeması ve `frame_{frame_index}_det_{detection_index}` kimliği
kullanılır. Crop oluştuysa JPEG'i `crops/frame_{frame_index:06d}_det_{detection_index:02d}.jpg`
olarak yazın; `perf_counter()` ile OCR süresini ölçüp `client.read_crop` çağırın.
Başarılı OCR yanıtını `raw_ocr/<detection_id>.json` altına UTF-8,
`ensure_ascii=False`, girintili JSON olarak yazın. `NimOcrError` durumunda
`ocr_raw_text`, `ocr_text` ve `ocr_confidence` alanları `None`; `error` alanı
hata metni olmalıdır. Crop yoksa OCR çağrısı yapılmamalı ve hata alanı `empty crop`
olarak korunmalıdır. Her tespit hem detection hem OCR JSONL dosyasına yazılır;
anotasyon listesine başarılı metin ya da `None` eklenir. Her kaynak kare, boş
anotasyon listesinde bile, `annotate_frame(frame, annotations)` sonrasında
writer'a yazılır; böylece kare sayısı kaynakla aynı kalır.

- [ ] **Step 4: Pipeline testlerini çalıştırıp başarılı olduklarını doğrulayın**

Run: `pytest tests/test_video_processing.py -v`

Expected: PASS.

- [ ] **Step 5: Tamamlanan pipeline'ı commit edin**

```bash
git add src/plate_ocr/processing/__init__.py src/plate_ocr/processing/video.py tests/test_video_processing.py
git commit -m "feat: process video with detection and OCR"
```

### Task 3: Komut Satırı Entegrasyonu ve Kullanım Dokümantasyonu

**Files:**
- Create: `src/plate_ocr/processing/cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_video_processing_cli.py`

**Interfaces:**
- Consumes: `--video`, `--model`, `--runs-dir`, `--base-url`, `--confidence`, `--image-size`, `--frame-stride`, `--timeout` CLI argümanları.
- Produces: `plate-ocr-process-video` console script'i; başarıda run dizini ve sayaçları stdout'a yazar, geçersiz dosya/donanım hatasında `SystemExit` ile açık mesaj verir.

- [ ] **Step 1: Varsayılanları ve girdi doğrulamasını kapsayan başarısız CLI testini yazın**

```python
def test_parse_arguments_uses_detection_and_nim_defaults() -> None:
    arguments = parse_arguments(["--video", "input.mp4", "--model", "model.onnx"])
    assert arguments.runs_dir == Path("runs")
    assert arguments.base_url == "http://127.0.0.1:8000"
    assert (arguments.confidence, arguments.image_size, arguments.frame_stride) == (0.35, 1280, 1)
    assert arguments.timeout == 30.0
```

- [ ] **Step 2: Testi başarısız olduğunu doğrulayarak çalıştırın**

Run: `pytest tests/test_video_processing_cli.py -v`

Expected: `ModuleNotFoundError: No module named 'plate_ocr.processing.cli'`.

- [ ] **Step 3: CLI, paket entry point'i ve README örneğini ekleyin**

```python
parser = argparse.ArgumentParser(
    description="Detect plates, read them with Nemotron OCR, and write an annotated video."
)
parser.add_argument("--video", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--timeout", type=float, default=30.0)
```

`pyproject.toml` içindeki `[project.scripts]` bölümüne
`plate-ocr-process-video = "plate_ocr.processing.cli:main"` ekleyin.
README'ye NIM hazır olduktan sonra kullanılacak tam komutu, çıktı dizini
içeriğini ve videonun sessiz olduğunu anlatan bir bölüm ekleyin.

- [ ] **Step 4: CLI ve tüm test paketini çalıştırın**

Run: `pytest tests/test_video_processing_cli.py -v && pytest -q`

Expected: tüm testler PASS.

- [ ] **Step 5: CLI ve dokümantasyon değişikliklerini commit edin**

```bash
git add src/plate_ocr/processing/cli.py pyproject.toml README.md tests/test_video_processing_cli.py
git commit -m "feat: expose annotated video command"
```

### Task 4: Gerçek Video ile Uçtan Uca Doğrulama

**Files:**
- Modify: `README.md` (yalnızca doğrulama komutu/durum bilgisi eksikse)

**Interfaces:**
- Consumes: Hazır NIM endpoint'i, `data/input/sample_plate_video.mp4`, `models/license-plate-finetune-v1l.onnx`.
- Produces: Yeni bir `runs/<run_id>/annotated.mp4` ve tam izlenebilir sonuçlar.

- [ ] **Step 1: NIM hazır durumunu doğrulayın**

Run: `curl --fail --silent http://127.0.0.1:8000/v1/health/ready`

Expected: `"ready":true` içeren JSON yanıtı.

- [ ] **Step 2: Gerçek videoda komutu çalıştırın**

Run:

```bash
plate-ocr-process-video \
  --video data/input/sample_plate_video.mp4 \
  --model models/license-plate-finetune-v1l.onnx
```

Expected: Yeni run dizini, kare/tespit/OCR sayaçları ve sıfır olmayan
`annotated.mp4` oluşur.

- [ ] **Step 3: Çıktı bütünlüğünü doğrulayın**

Run:

```bash
ffprobe -v error -show_entries format=duration:stream=codec_type,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 runs/<run_id>/annotated.mp4
```

Expected: Bir video stream'i, ses stream'i olmaması, `1280x720`, `25/1` FPS
ve sıfırdan büyük süre. `detections.jsonl` ile `ocr_results.jsonl` satır
sayılarını ve `raw_ocr/` altındaki başarılı OCR yanıtlarını da karşılaştırın.

- [ ] **Step 4: Git durumunu doğrulayın ve yalnızca kaynak değişikliklerini commit edin**

Run: `git status --short && git log --oneline -4`

Expected: `runs/` Git tarafından dışlanmış; gerçek çalışma çıktıları stage
alanına girmemiş olmalıdır.
