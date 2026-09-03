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
