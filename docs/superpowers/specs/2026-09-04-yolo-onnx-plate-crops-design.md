# YOLOv11 ONNX ile Plaka Tespiti ve Crop Tasarımı

## Amaç

`data/input/sample_plate_video.mp4` videosundaki plakaları YOLOv11 Large
ONNX modeliyle tespit etmek, her tespiti JPEG crop olarak kaydetmek ve
izlenebilir JSONL kaydı üretmek.

## Model

Model kaynağı Hugging Face'teki
`morsetechlab/yolov11-license-plate-detection` deposudur. Kullanılacak dosya
`license-plate-finetune-v1l.onnx` dosyasıdır. İndirme, model deposunun
doğrulanmış `251a30d` commit'ine sabitlenir ve beklenen SHA-256 değeri
`5efdfbe4909bfa6c895bed48676b7de695bf71788932e095e7bc74b8b52b75d8` olarak
kontrol edilir.

ONNX seçimi, modelin Large varyantını kullanırken `.pt` dosyalarının pickle
tabanlı yükleme riskinden kaçınır. Model AGPL-3.0 lisanslıdır; bu PoC yerel
değerlendirme içindir ve dağıtım öncesi lisans uyumluluğu ayrıca incelenir.

## Bileşenler

- `scripts/download_model.py`: Modeli sabit Hugging Face revision'ından
  `models/license-plate-finetune-v1l.onnx` yoluna indirir; indirme sonrası
  SHA-256 doğrulaması başarısızsa dosyayı reddeder.
- `src/plate_ocr/config.py`: Video, model, eşik, görüntü boyutu, kare atlama
  ve çıktı ayarlarını doğrular.
- `src/plate_ocr/detection/yolo.py`: Ultralytics üzerinden ONNX modeli yükler
  ve BGR OpenCV karesinden bağımsız tespit kayıtları üretir.
- `src/plate_ocr/pipeline.py`: Video karelerini sırayla okur, tespitleri
  crop'lara dönüştürür ve JSONL kaydını yazar.
- `src/plate_ocr/cli.py`: Kullanıcının pipeline'ı komut satırından çalıştırdığı
  giriş noktasıdır.

## Çalışma Akışı

1. Kullanıcı modeli indirir; ağırlık `models/` altında kalır ve Git'e eklenmez.
2. CLI, video ve model yollarını doğrular; varsayılanlar `confidence=0.35`,
   `image_size=1280` ve `frame_stride=1` olur.
3. Her seçilen kare YOLO'ya gönderilir.
4. Her bbox görüntü sınırlarına sıkıştırılır. Sıfır alanlı bbox veya boş crop
   kaydedilmez; olay kaydına hata yazılır ve sonraki tespit/kare işlenir.
5. Geçerli crop, `runs/<run_id>/crops/frame_<frame>_det_<index>.jpg` olarak
   kaydedilir.
6. Her geçerli tespit için `detections.jsonl` içine `video_id`, `frame_index`,
   `timestamp_ms`, `detection_id`, `bbox_xyxy`, `det_confidence` ve
   `crop_path` yazılır.

İlk kapsam yalnızca tespit ve crop üretimidir: OCR, takip, anotasyonlu video
ve CSV raporu sonraki fazlarda eklenecektir.

## Hata Yönetimi

- Video açılamıyorsa CLI açıklayıcı hata ile başarısız olur.
- Model bulunamaz veya SHA-256 eşleşmezse çalıştırma başlamaz.
- Ultralytics veya ONNX çalışma zamanı hatası, kare numarası bağlamıyla
  raporlanır.
- Yazılamayan çıktı klasörü ve JPEG yazma hatası başarısız çalıştırma olarak
  bildirilir.

## Testler

- Yapılandırma varsayılanları ve geçersiz parametreler birim testleriyle
  doğrulanır.
- Sahte algılayıcıyla çalışan küçük video entegrasyon testi, crop dosyası ve
  JSONL alanlarını doğrular; gerçek modele/GPU'ya bağlı değildir.
- Crop sınırlandırma testi, görüntü dışına taşan bbox'ların geçerli crop
  ürettiğini ve sıfır alanlı bbox'ların kaydedilmediğini doğrular.

## Başarı Kriteri

Komut, örnek videoyu baştan sona işleyerek `runs/<run_id>/crops/` altında
okunabilir plaka crop'ları ve bunlarla bire bir eşleşen JSONL tespit kayıtları
oluşturur.
