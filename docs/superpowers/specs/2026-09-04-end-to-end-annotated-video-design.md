# Uçtan Uca Anotasyonlu Video Tasarımı

## Amaç

Bir kaynak videoyu her çalıştırmada YOLO ONNX plaka tespiti ve Nemotron OCR
NIM ile baştan işlemek; crop'lar ve yapılandırılmış sonuçlarla birlikte
plaka kutuları ve OCR metinleri işlenmiş sessiz bir MP4 üretmek.

## Komut ve Girdiler

Yeni `plate-ocr-process-video` komutu aşağıdaki girdileri alır:

```text
--video data/input/sample_plate_video.mp4
--model models/license-plate-finetune-v1l.onnx
--runs-dir runs
--base-url http://127.0.0.1:8000
--confidence 0.35
--image-size 1280
--frame-stride 1
--timeout 30
```

YOLO ve OCR ayarlarının varsayılanları mevcut ayrı `plate-ocr-detect` ve
`plate-ocr-ocr` komutlarıyla aynıdır. Komut başlamadan kaynak video ve model
dosyasının mevcut olduğunu denetler. OCR NIM erişilemezse çalıştırma net bir
hata kaydı üretir; eksik, geçersiz ya da erişilemeyen OCR sonucu yalnızca
ilgili tespitin kaydına yazılır ve sonraki kareler işlenmeye devam eder.

## Akış ve Çıktılar

Komut yeni, UTC zaman damgalı bir `runs/<run_id>/` dizini oluşturur ve kaynak
videoyu kare kare bir kez okur. İşlenen her karede:

1. YOLO tespitleri üretilir ve geçerli crop'lar `crops/` içine yazılır.
2. Her crop NIM'e gönderilir; ham başarılı yanıt
   `raw_ocr/<detection_id>.json` içine yazılır.
3. Tespit ve OCR alanları birleştirilerek `ocr_results.jsonl` içine tek
   kayıt olarak yazılır.
4. Tespit kutusu ve OCR etiketi kaynak kare üzerine çizilir; kare
   `annotated.mp4` çıktısına eklenir.

`detections.jsonl` de her tespit için ayrıca yazılır ve mevcut komutun şeması
ile uyumludur. `ocr_results.jsonl`, mevcut OCR replay çıktısındaki alanları
korur: tespit alanları ile `ocr_raw_text`, `ocr_text`, `ocr_confidence`,
`ocr_latency_ms` ve `error`.

Çıktı video sessizdir; kaynak videonun çözünürlüğünü ve FPS değerini korur.
Video yazımı OpenCV'nin MP4 desteğiyle yapılır. Ses akışı kopyalanmaz.

## Anotasyon Kuralları

- Her geçerli tespit, kırpılmış koordinatlarla yeşil bir dikdörtgenle çizilir.
- Etiket, kutunun üstünde kontrast bir arka plan üzerinde normalize edilmiş
  `ocr_text` olarak gösterilir.
- OCR metni boşsa, servis hatası varsa veya crop üretilememişse etiket
  `OKUNAMADI` olur.
- Etiket çizimi kare sınırları içinde tutulur; koordinat veya metin çizim
  hatası video akışını kesmez.
- Bir karede birden çok plaka varsa her tespit bağımsız olarak işlenir ve
  çizilir.

## Bileşen Sınırları

- `plate_ocr.processing` uçtan uca akışı yönetir: run dizini, kare okuma,
  tespit, crop, OCR çağrısı, JSONL ve video yazımı.
- Yeni video anotasyon modülü yalnızca BGR kare ile tespit/OCR verisini alır
  ve çizimi yapar; dosya sistemi ya da NIM istemcisi bilmez.
- Yeni CLI modülü argümanları ayrıştırır, mevcut YOLO ve NIM adaptörlerini
  oluşturur ve sonuç sayaçlarını yazdırır.
- Mevcut bağımsız detection ve OCR replay komutları değişmeden kalır.

## Hata Yönetimi ve Testler

- Kaynak video açılamazsa, FPS sıfır/geçersizse veya MP4 yazıcısı açılamazsa
  açık bir hata üretilir ve video çıktısı başarılı sayılmaz.
- OCR servis hataları ayrı kayıt olarak saklanır; aynı karedeki diğer
  tespitler ve sonraki kareler devam eder.
- Birim testleri kutu sınırlandırmayı, metin geri dönüşünü ve anotasyonun
  kare piksellerini değiştirdiğini doğrular.
- Sahte detector ve sahte OCR istemcisiyle yapılan entegrasyon testi, küçük
  yapay videodan `detections.jsonl`, `ocr_results.jsonl`, crop, ham OCR
  yanıtı ve okunabilir, sessiz `annotated.mp4` üretildiğini doğrular.

## Başarı Kriteri

NIM hazır durumdayken tek komut yeni bir run dizini oluşturur. Bu dizin
izlenebilir tespit/OCR kayıtlarını ve kaynak kare sayısıyla aynı sayıda,
kutular ile OCR etiketleri çizilmiş sessiz `annotated.mp4` dosyasını içerir.
