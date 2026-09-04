# Nemotron OCR v2 NIM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Yerel Nemotron OCR v2 NIM servisini güvenli şekilde çalıştırmak ve mevcut plaka crop run'larını OCR sonuçlarına dönüştürmek.

**Architecture:** Docker Compose, NIM'i yalnızca localhost üzerinde sabit imaj etiketi ve kalıcı cache ile çalıştırır. Python istemcisi NIM HTTP şemasını `NimOcrClient` arkasında saklar; OCR runner mevcut detection JSONL'ini değiştirmeden ham cevaplar ve yeni OCR JSONL dosyası üretir.

**Tech Stack:** Docker Compose, NVIDIA Container Toolkit, Nemotron OCR v2 NIM 2.0, Python 3.11, httpx, pytest.

## Global Constraints

- NIM imajı `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0` olarak sabitlenir.
- NIM yalnızca `127.0.0.1:8000` üzerinde yayınlanır ve GPU `0` kullanır.
- `NGC_API_KEY` yalnızca dışlanan `.env` içinde olur; `.env.example` sır içermez.
- İstemci `/v1/ocr` endpoint'ini JPEG base64 data URL ile çağırır.
- İlk sürüm crop başına tek HTTP isteği yapar; OCR sonuçları `ocr_results.jsonl` ve `raw_ocr/` altına yazılır.
- `detections.jsonl` değiştirilmez.

---

### Task 1: NIM Compose yapılandırması

**Files:**
- Create: `.env`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Produces: `docker compose up -d nemotron-ocr` ile başlayan, `http://127.0.0.1:8000` taban URL'sine sahip OCR servisi.

- [ ] **Step 1: Compose doğrulama girdisini hazırlayın**

`.env.example` aşağıdaki sır içermeyen değerleri taşımalıdır:

```dotenv
NGC_API_KEY=
NIM_IMAGE=nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0
NIM_CACHE_DIR=.nim-cache
OCR_BASE_URL=http://127.0.0.1:8000
```

- [ ] **Step 2: Docker Compose kuralını yazın ve doğrulayın**

`nemotron-ocr` servisi `gpus: all`, `shm_size: 16gb`, NGC model indirme değişkenleri ve `127.0.0.1:8000:8000` port eşlemesi kullanır. Cache alt dizinleri `/opt/cache` ve `/model` içine bağlanır.

Run: `docker compose --env-file .env.example config -q`

Expected: exit code 0.

- [ ] **Step 3: Yerel cache'i dışlayın ve kullanım belgesi ekleyin**

```gitignore
.nim-cache/
```

README, NGC lisans kabulü, `.env` anahtar girişi, `docker login`, başlatma,
log, durdurma ve hazır olma kontrolü komutlarını içerir.

- [ ] **Step 4: Yapılandırmayı commit edin**

```bash
git add .gitignore .env.example docker-compose.yml README.md
git commit -m "build: add local Nemotron OCR NIM service"
```

### Task 2: NIM HTTP istemcisi ve OCR metin işleme

**Files:**
- Modify: `pyproject.toml`
- Create: `src/plate_ocr/ocr/__init__.py`
- Create: `src/plate_ocr/ocr/types.py`
- Create: `src/plate_ocr/ocr/normalize.py`
- Create: `src/plate_ocr/ocr/nim_client.py`
- Create: `tests/test_ocr_normalize.py`
- Create: `tests/test_nim_client.py`

**Interfaces:**
- Produces: `OcrResult(raw_text: str, text: str, confidence: float | None, raw_response: dict[str, object])`; `normalize_plate_text(value: str) -> str`; `NimOcrClient.read_crop(path: Path) -> OcrResult`.

- [ ] **Step 1: Write failing text normalization tests**

```python
def test_normalize_plate_text_preserves_raw_meaning_without_substitution():
    assert normalize_plate_text(" 34 ab-123 ") == "34AB123"

def test_normalize_plate_text_does_not_substitute_ambiguous_characters():
    assert normalize_plate_text("O1 S5") == "O1S5"
```

- [ ] **Step 2: Run normalization test to verify RED**

Run: `python -m pytest tests/test_ocr_normalize.py -v`

Expected: FAIL because `plate_ocr.ocr.normalize` does not exist.

- [ ] **Step 3: Implement minimal normalization**

Uppercase text and retain only Unicode alphanumeric characters. Do not map
`O/0`, `I/1`, `S/5` or `B/8`.

- [ ] **Step 4: Run normalization test to verify GREEN**

Run: `python -m pytest tests/test_ocr_normalize.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing NIM client tests with `httpx.MockTransport`**

```python
def test_read_crop_posts_jpeg_data_url_and_parses_word_detections(tmp_path):
    client = NimOcrClient("http://nim.test", transport=mock_transport)
    result = client.read_crop(write_jpeg(tmp_path / "crop.jpg"))
    assert result.raw_text == "34 AB 123"
    assert result.text == "34AB123"
    assert result.confidence == 0.98

def test_read_crop_raises_nim_error_for_http_failure(tmp_path):
    with pytest.raises(NimOcrError, match="503"):
        NimOcrClient("http://nim.test", transport=service_unavailable).read_crop(write_jpeg(tmp_path / "crop.jpg"))
```

- [ ] **Step 6: Run client test to verify RED; implement client; verify GREEN**

Run: `python -m pytest tests/test_nim_client.py -v`

Expected before implementation: FAIL importing `plate_ocr.ocr.nim_client`; expected after implementation: PASS.

Use `httpx.Client(timeout=30.0)`, POST `${base_url}/v1/ocr`, `merge_levels=["word"]`, and convert successful `text_prediction` entries to raw text and mean confidence.

- [ ] **Step 7: Add `httpx` dependency and commit Task 2**

```bash
git add pyproject.toml src/plate_ocr/ocr tests/test_ocr_normalize.py tests/test_nim_client.py
git commit -m "feat: add Nemotron OCR client"
```

### Task 3: OCR run replay and CLI

**Files:**
- Create: `src/plate_ocr/ocr/runner.py`
- Create: `src/plate_ocr/ocr/cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `tests/test_ocr_runner.py`
- Create: `tests/test_ocr_cli.py`

**Interfaces:**
- Consumes: a prior run's `detections.jsonl`, crop files and `NimOcrClient` interface.
- Produces: `OcrRunResult(run_dir: Path, processed_count: int, success_count: int, error_count: int)` from `run_ocr(run_dir: Path, client: CropOcrClient)`.

- [ ] **Step 1: Write a failing fake-client integration test**

```python
def test_run_ocr_writes_raw_response_and_results_for_every_crop(tmp_path):
    result = run_ocr(detection_run_with_two_crops(tmp_path), FakeOcrClient())
    records = read_jsonl(result.run_dir / "ocr_results.jsonl")
    assert result.success_count == 2
    assert [record["ocr_text"] for record in records] == ["34AB123", "06XYZ99"]
    assert all((result.run_dir / "raw_ocr" / f"{record['detection_id']}.json").is_file() for record in records)
```

- [ ] **Step 2: Run runner test to verify RED**

Run: `python -m pytest tests/test_ocr_runner.py -v`

Expected: FAIL because `plate_ocr.ocr.runner` does not exist.

- [ ] **Step 3: Implement OCR replay runner**

Read each detection record with a non-null crop path. On success write the
raw response as pretty JSON and an OCR result record. On `NimOcrError`, write
an OCR result record whose OCR fields are null and whose `error` is the
exception message; continue processing following detections.

- [ ] **Step 4: Run runner test to verify GREEN**

Run: `python -m pytest tests/test_ocr_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Add and test `plate-ocr-ocr` CLI**

```bash
plate-ocr-ocr --run runs/20260904T073711618886Z --base-url http://127.0.0.1:8000
```

The CLI validates the run directory and `detections.jsonl`, prints processed,
success, and error counters, and exits non-zero for missing input.

- [ ] **Step 6: Run the full automated suite and commit Task 3**

Run: `python -m pytest -v`

Expected: PASS.

```bash
git add src/plate_ocr/ocr tests/test_ocr_runner.py tests/test_ocr_cli.py pyproject.toml README.md
git commit -m "feat: add OCR replay command"
```

### Task 4: NIM launch and real crop verification

**Files:**
- Modify: ignored `.env` (user supplies only `NGC_API_KEY` value)
- Create: ignored `.nim-cache/`
- Create: ignored `runs/<run_id>/raw_ocr/` and `ocr_results.jsonl`

**Interfaces:**
- Consumes: Tasks 1-3 and an NGC account whose OCR v2 terms were accepted.
- Produces: ready local NIM and OCR records for the existing sample-video run.

- [ ] **Step 1: User supplies the NGC secret locally**

Set only this line in `.env`:

```dotenv
NGC_API_KEY=your-personal-ngc-key
```

- [ ] **Step 2: Authenticate and launch NIM**

```bash
set -a; . ./.env; set +a
printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker compose up -d nemotron-ocr
```

- [ ] **Step 3: Wait for readiness**

Run: `curl --fail --retry 60 --retry-delay 5 http://127.0.0.1:8000/v1/health/ready`

Expected: JSON with `"ready":true`.

- [ ] **Step 4: Run OCR against the existing detection run**

Run: `plate-ocr-ocr --run runs/20260904T073711618886Z --base-url http://127.0.0.1:8000`

Expected: OCR counters and `ocr_results.jsonl` output.

- [ ] **Step 5: Verify OCR output integrity**

Run: `python -c "import json; from pathlib import Path; p=Path('runs/20260904T073711618886Z/ocr_results.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines()]; print(len(rows), sum(x['error'] is None for x in rows))"`

Expected: one OCR row per crop and a non-negative success count.
