# Nemotron OCR v2 NIM Kurulumu ve OCR Entegrasyonu Tasarımı

## Amaç

NVIDIA Nemotron OCR v2 NIM servisini bu makinede GPU 0 üzerinde çalıştırmak ve
mevcut plaka crop çıktılarını OCR sonuçlarıyla ilişkilendirmek.

## NIM Servisi

Servis Docker Compose ile çalıştırılır. İmaj etiketi
`nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0` olarak sabitlenir. Servis GPU 0,
16 GB shared memory ve kalıcı `.nim-cache/` dizini kullanır. Host portu
yalnızca `127.0.0.1:8000` üzerinde açılır; NIM HTTP API kendi başına kimlik
doğrulaması sağlamadığından genel ağa açılmaz.

`.env` Git tarafından dışlanır ve yalnızca gerçek `NGC_API_KEY` değerini,
sabit imaj adını, cache yolunu ve yerel OCR taban URL'sini içerir.
`.env.example` aynı değişkenleri boş anahtarla belgelendirir. Kullanıcı
NGC anahtarını yalnızca kendi `.env` dosyasına girer; anahtar kaynak koda,
loglara ve Git geçmişine yazılmaz.

Başlatma akışı NGC registry girişini, lisans koşullarının kabulünü,
`docker compose up -d` çalıştırılmasını ve
`GET http://127.0.0.1:8000/v1/health/ready` sağlık kontrolünü içerir.

## OCR İstemcisi

`src/plate_ocr/ocr/nim_client.py`, crop JPEG dosyasını base64 data URL'e
dönüştürür ve `${OCR_BASE_URL}/v1/ocr` adresine aşağıdaki biçimde gönderir:

```json
{
  "input": [{"type": "image_url", "url": "data:image/jpeg;base64,..."}],
  "merge_levels": ["word"]
}
```

İstemci `httpx` kullanır; bağlantı, zaman aşımı ve HTTP hatalarını açık hata
kaydı olarak döndürür. İlk sürüm her crop için tek istek yapar. Servis
cevabındaki `text_detections` sırasını korur, metinleri boşlukla birleştirir
ve sağlanmışsa metin güvenlerini saklar. OCR metni yalnızca büyük harfe
dönüştürülüp boşluk/plaka dışı karakterler temizlenerek normalize edilir;
ham metin ve ham API cevabı korunur.

## Çıktı Akışı

Yeni `plate-ocr-ocr --run <run_dir>` komutu mevcut bir detection run'ındaki
`detections.jsonl` kaydını ve crop yollarını okur. Her geçerli crop için:

1. `raw_ocr/<detection_id>.json` içinde ham HTTP cevabını yazar.
2. `ocr_results.jsonl` içinde detection alanlarıyla birlikte `ocr_raw_text`,
   `ocr_text`, `ocr_confidence`, `ocr_latency_ms` ve `error` alanlarını yazar.

Mevcut `detections.jsonl` değişmez; böylece OCR parser'ı veya servis sürümü
değiştiğinde crop'lar üzerinden yeniden OCR çalıştırılabilir.

## Hata Yönetimi ve Testler

- Eksik `.env` veya `NGC_API_KEY` Compose başlamadan önce açık hata üretir.
- Sağlık kontrolü hazır olmayan servisi başarılı kabul etmez.
- HTTP istemcisi HTTP hata kodu, zaman aşımı ve bozuk JSON yanıtını ayrı hata
  kayıtları olarak üretir; diğer crop'ları işlemeye devam eder.
- Birim testleri, base64 istek gövdesini, cevap parser'ını ve normalizasyonu
  doğrular.
- Sahte OCR istemcisi kullanılan entegrasyon testi, iki crop için iki ham
  cevap dosyası ve doğru `ocr_results.jsonl` kayıtlarını doğrular.

## Başarı Kriteri

Yerel NIM sağlık endpoint'i hazır yanıtı verir. Bir mevcut crop run'ı OCR
komutuyla işlendiğinde her crop için ya izlenebilir OCR sonucu ve ham cevap
dosyası ya da açık hata kaydı bulunur.
