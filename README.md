# Araç Plakası Tespit ve OCR

Bu proje, kayıtlı araç videolarında YOLO ile plaka tespiti yapıp NVIDIA
Nemotron OCR v2 ile metin okuma için hazırlanmış bir PoC'tur. Mimari ve
pipeline hedefleri için [PLAN.md](PLAN.md) dosyasına bakın.

## Video dosyalarını GitHub ile aktarma

Kaynak videolar Git LFS ile sürümlenir. `data/input/` altındaki `.mp4`,
`.mov`, `.avi` ve `.mkv` dosyalarını normal `git add`, `git commit` ve
`git push` komutlarıyla gönderin. `runs/` altındaki üretilen videolar,
OCR kırpımları, model ağırlıkları ve gizli dosyalar gönderilmez.

### İlk bilgisayar: video ekleme ve gönderme

Git LFS'yi bilgisayarda bir kez kurup etkinleştirin. Ardından videoyu proje
altına kopyalayıp commit edin:

```bash
git lfs install
mkdir -p data/input
cp /mutlak/yol/video.mp4 data/input/
git add data/input/video.mp4
git commit -m "data: add input video"
git push origin main
```

Yüklemeden önce LFS'in videoyu izlediğini doğrulayabilirsiniz:

```bash
git lfs ls-files
```

### İkinci bilgisayar: video indirme

İkinci bilgisayarda Git ve Git LFS kurulu olmalıdır. Depoyu klonlayıp LFS
içeriğini indirin:

```bash
git lfs install
git clone https://github.com/llakyoll/ocr_with_yolo_nemotron.git
cd ocr_with_yolo_nemotron
git lfs pull
```

Sonraki değişikliklerde:

```bash
git pull
git lfs pull
```

GitHub LFS depolama ve bant genişliği kotasına tabidir. Büyük veya sık
değişen ham veri setleri için harici obje depolama kullanın.

## YOLO ile plaka tespiti ve crop üretimi

İlk çalıştırmadan önce proje bağımlılıklarını editable olarak kurun:

```bash
python -m pip install -e ".[dev]"
```

Large YOLOv11 ONNX ağırlığını indirip SHA-256 değerini doğrulayın:

```bash
python scripts/download_model.py
```

Örnek videodaki tüm kareleri işleyin:

```bash
python -m plate_ocr.cli \
  --video data/input/sample_plate_video.mp4 \
  --model models/license-plate-finetune-v1l.onnx
```

Her çalıştırma `runs/<UTC_run_id>/` altında `crops/` ve
`detections.jsonl` üretir. Çıktılar Git'e eklenmez. Hızı denemek için
`--frame-stride 5`, tespit eşiğini değiştirmek için `--confidence 0.50`
kullanabilirsiniz.

## Yerel Nemotron OCR v2 NIM

NIM, yalnızca bu makinenin erişebildiği `127.0.0.1:8000` adresinde Docker
Compose ile çalışır. Önce NGC'de Nemotron OCR v2 koşullarını kabul edin,
ardından `.env` içindeki tek boş değere kişisel anahtarınızı yazın:

```dotenv
NGC_API_KEY=your-personal-ngc-api-key
```

Anahtarı sohbete veya Git'e göndermeyin. İmajı çekmek için giriş yapın ve
servisi başlatın:

```bash
set -a; . ./.env; set +a
printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker compose up -d nemotron-ocr
curl --fail --retry 60 --retry-delay 5 http://127.0.0.1:8000/v1/health/ready
```

Çalışma günlükleri ve durdurma komutları:

```bash
docker compose logs -f nemotron-ocr
docker compose stop nemotron-ocr
```

Model ağırlıkları `.nim-cache/` altında kalır ve Git'e eklenmez.
