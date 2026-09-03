# Arac Plakasi Tespit ve OCR Projesi

## 1. Amac

Bu projenin ilk amaci, kayitli bir video uzerindeki arac plakalarini YOLO ile tespit etmek ve tespit edilen plaka goruntulerini yerel olarak calistirilan NVIDIA Nemotron OCR v2 servisi ile okumaktir.

Ilk asamada sistemi erken optimize etmek yerine uctan uca calisan ve olculebilir bir temel hat (baseline) kurulacaktir. Bu hattin ciktilari incelendikten sonra tespit, goruntu on isleme, takip, OCR ve sonuc birlestirme adimlari ayri ayri iyilestirilecektir.

## 2. Ilk Asamanin Kapsami

Ilk PoC asagidaki akisla sinirlidir:

1. Bir video dosyasini kare kare okumak.
2. YOLO ile her karede plaka tespiti yapmak.
3. Tespit edilen plaka alanlarini kirpmak ve kaydetmek.
4. Kirpilan plakalari Nemotron OCR v2 servisine gondermek.
5. Ham OCR metnini, guven skorlarini ve ilgili goruntu bilgilerini kaydetmek.
6. Islenmis videoyu tespit kutulari ve OCR sonucuyla birlikte uretmek.
7. Sonuclari incelemeye uygun CSV/JSON raporuna donusturmek.

Ilk PoC kapsaminda canli kamera, coklu kamera, web arayuzu, veritabani, alarm sistemi, otomatik kapi acma ve uretim olcekleme bulunmayacaktir.

## 3. Onerilen Mimari

```text
Video dosyasi
    |
    v
OpenCV video okuyucu
    |
    v
YOLO plaka tespiti
    |
    +----> Tespit kutusu ve skor
    |
    v
Plaka kirpimi ve temel goruntu hazirlama
    |
    v
Nemotron OCR v2 NIM servisi
    |
    v
Ham OCR cevabi
    |
    v
Normalizasyon ve plaka bicimi kontrolu
    |
    +----> JSON / CSV olay kaydi
    |
    +----> Isaretlenmis cikti videosu
    |
    +----> Inceleme icin plaka kirpimlari
```

### Calisma ortami

- **Gelisme makinesi:** Kod gelistirme, video akisinin kurulmasi ve gerekirse YOLO testi.
- **Guclu GPU sunucusu:** Nemotron OCR v2 NIM container'i ve tercihen tam pipeline testi.
- **Baglanti modeli:** Uygulama, yapilandirilabilir bir `OCR_BASE_URL` uzerinden Nemotron servisine HTTP istegi gonderir.
- **Varsayilan guvenlik:** Servis genel internete dogrudan acilmaz. Ayni ag, VPN veya SSH tunnel kullanilir. API anahtari ve kimlik bilgileri repoya yazilmaz.

YOLO ve OCR ayni guclu sunucuda calisabiliyorsa goruntuleri ag uzerinden tasimamak daha saglikli olacaktir. Gelistirme makinesinden uzaktaki OCR servisine baglanma secenegi yine korunacaktir.

## 4. Teknoloji Secimleri

- Python 3.11
- OpenCV: video okuma, kirpma ve cikti videosu
- Ultralytics YOLO: plaka tespiti
- NVIDIA Nemotron OCR v2 NIM: OCR servisi
- HTTP istemcisi: `httpx` veya `requests`
- Pydantic: ayar ve sonuc semalari
- PyYAML: calisma konfigurasyonu
- pytest: birim ve entegrasyon testleri
- Docker Compose: uygulama bagimliliklari; Nemotron NIM gerekirse ayri calistirilir

YOLO modelinin surumu ve agirlik dosyasi konfigurasyonda acikca belirtilmeli, deney boyunca degistirilmemelidir. Nemotron icin `latest` yerine dogrulanmis container etiketi sabitlenmelidir.

## 5. Veri Akisi ve Kayit Formati

Her tespit icin en az su alanlar kaydedilecektir:

| Alan | Aciklama |
| --- | --- |
| `video_id` | Kaynak videonun benzersiz kimligi |
| `frame_index` | Tespitin bulundugu kare |
| `timestamp_ms` | Video icindeki zaman |
| `detection_id` | Tespitin benzersiz kimligi |
| `bbox_xyxy` | Plaka sinir kutusu |
| `det_confidence` | YOLO guven skoru |
| `crop_path` | Kaydedilen plaka kirpimi |
| `ocr_raw_text` | Nemotron'un ham metni |
| `ocr_text` | Normalize edilmis metin |
| `ocr_confidence` | API tarafindan saglaniyorsa OCR guveni |
| `format_valid` | Plaka bicim kontrolu sonucu |
| `ocr_latency_ms` | OCR istek suresi |
| `error` | Varsa hata veya zaman asimi bilgisi |

Ham Nemotron cevabi ayrica saklanacaktir. Boylece parser degistiginde OCR servisini yeniden calistirmadan sonuclar tekrar islenebilir.

## 6. Plaka Metni Isleme

Ilk asamada OCR sonucuna agresif duzeltme uygulanmayacaktir. Asagidaki islemler yeterlidir:

- Metni buyuk harfe donusturmek.
- Bosluk ve plaka disi karakterleri temizlemek.
- Ham sonucu her zaman korumak.
- Turkiye plakasi bicimine uygunlugu ayri bir alan olarak raporlamak.

Bicim kontrolu OCR sonucunu sessizce degistirmemelidir. `O/0`, `I/1`, `S/5`, `B/8` gibi donusumler ancak gercek hata verisi incelendikten sonra ve gerekceli kurallarla eklenmelidir.

## 7. Uygulama Asamalari

### Faz 0 - Ortam ve erisim dogrulamasi

- Guclu sunucuda guncel NVIDIA surucusu, Docker ve NVIDIA Container Toolkit'i dogrulamak.
- NGC API anahtarini sunucuda secret/environment olarak tanimlamak.
- Nemotron OCR v2 container'ini sabit bir surum etiketiyle calistirmak.
- Saglik kontrolu ve tek plaka goruntusuyle API testini tamamlamak.
- Kullanilan NIM surumunde endpoint ve cevap semasini kaydetmek.

**Cikis kriteri:** Ornek bir goruntu yerel NIM API'sine gonderilebiliyor ve cevap kaydedilebiliyor.

### Faz 1 - YOLO video pipeline'i

- Video girisi ve cikti ayarlarini olusturmak.
- Plaka tespit modelini yuklemek.
- Karelerde tespit yapmak ve plaka kirpimlarini kaydetmek.
- Tespit kutularini cikti videosuna cizmek.
- Bos kare, bozuk video ve GPU bellek hatalarini yonetmek.

**Cikis kriteri:** Video bastan sona isleniyor; tespitler, kirpimlar ve isaretlenmis video uretiliyor.

### Faz 2 - Nemotron entegrasyonu

- OCR istemcisini NIM endpoint'inden bagimsiz bir arayuz arkasinda kurmak.
- JPEG/PNG kodlama, zaman asimi, sinirli yeniden deneme ve hata kaydi eklemek.
- OCR cevabini normalize edilmis sonuc semasina cevirmek.
- Ham API cevabini saklamak.
- Eszamanlilik ve batch ayarlarini konfigurasyona almak; ilk deneyi tekli isteklerle yapmak.

**Cikis kriteri:** Her uygun plaka kirpimi icin OCR sonucu veya acik bir hata kaydi bulunuyor.

### Faz 3 - Raporlama ve ilk degerlendirme

- JSONL ve CSV raporu olusturmak.
- OCR metnini cikti videosunda gostermek.
- Rastgele ornekler yerine sabit bir degerlendirme videosu kullanmak.
- Sonuclari elle dogrulanmis gercek plaka metinleriyle karsilastirmak.
- Hatalari tespit, kirpma, goruntu kalitesi, OCR ve bicimlendirme olarak siniflandirmak.

**Cikis kriteri:** Basari oranlari ve hata ornekleri tekrar uretilebilir bir raporda gorulebiliyor.

### Faz 4 - Sonuca gore iyilestirme

Ilk rapordan sonra yalnizca olculen probleme yonelik iyilestirme secilecektir:

- Tespit kaciriliyorsa YOLO modeli/verisi veya giris cozunurlugu.
- Plaka cok kucuk ya da egikse kirpma payi, perspektif duzeltme veya super-resolution.
- Ayni plaka tekrar tekrar okunuyorsa ByteTrack ve kareler arasi birlestirme.
- OCR karakter karistiriyorsa goruntu on isleme, coklu kirpim oylamasi veya plaka odakli alternatif OCR.
- Gecikme yuksekse en iyi kare secimi, batch ve asenkron istekler.

## 8. Ilk Deney Protokolu

Ilk deneyde ayarlar sabit tutulacaktir:

1. Temsili tek bir video secilir ve dosya hash'i kaydedilir.
2. YOLO agirligi, NIM container surumu ve konfigurasyon dosyasi rapora eklenir.
3. Video once varsayilan ayarlarla bir kez islenir.
4. Tespit edilen plakalardan elle kontrol edilebilir bir ornek kumesi olusturulur.
5. Gercek plaka metinleri ayri bir referans dosyasina yazilir.
6. Sonuclar ayni referans uzerinden hesaplanir.

Ayni plakaya ait ard arda kareler ilk raporda ayri OCR denemeleri olarak tutulabilir; arac bazli nihai basari ayrica hesaplanmalidir. Bu ayrim, kare bazli gurultunun gercek sistem performansini gizlemesini engeller.

## 9. Basari Metrikleri

Asagidaki metrikler birlikte raporlanacaktir:

- **Detection recall:** Gorunen plakalarin ne kadari tespit edildi?
- **Detection precision:** Tespitlerin ne kadari gercek plaka?
- **Exact plate accuracy:** Plakanin tamami kac ornekte dogru okundu?
- **Character accuracy:** Toplam karakterlerin ne kadari dogru?
- **Track/vehicle accuracy:** Bir araca ait karelerden en az birinde tam dogru sonuc elde edildi mi?
- **End-to-end success:** Gorunen bir plaka tespit edilip tam dogru okunabildi mi?
- **Latency:** YOLO, OCR ve toplam sure; ortalama ile birlikte p50/p95.
- **Failure rate:** Zaman asimi, servis hatasi ve islenemeyen kirpim sayisi.

Ilk PoC icin kesin bir yuzdeyi pesinen basari esigi yapmak yerine baseline olculmelidir. Bununla birlikte teknik kabul icin pipeline'in videoyu hatasiz tamamlamasi, tum tespitlerin izlenebilir kayda sahip olmasi ve ayni konfigurasyonla tekrar uretilebilir sonuc vermesi zorunludur.

## 10. Onerilen Proje Yapisi

```text
ocr_project/
├── PLAN.md
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   └── baseline.yaml
├── src/plate_ocr/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── detection/
│   ├── ocr/
│   ├── processing/
│   └── reporting/
├── tests/
├── scripts/
│   └── check_nim.py
├── data/
│   ├── input/
│   ├── ground_truth/
│   └── samples/
└── runs/
    └── <run_id>/
        ├── config.yaml
        ├── detections.jsonl
        ├── results.csv
        ├── crops/
        ├── raw_ocr/
        └── annotated.mp4
```

`runs/`, model agirliklari ve secret dosyalari Git'e eklenmemelidir. Kaynak
videolar `data/input/` altinda Git LFS ile surumlenebilir; GitHub LFS depolama
ve bant genisligi kotalari izlenmeli, buyuk veya sik degisen ham veri setleri
gerektiginde harici obje depolamaya tasinmalidir.

## 11. Yapilandirma Taslagi

```yaml
input:
  video_path: data/input/sample.mp4

detection:
  model_path: models/license_plate_yolo.pt
  confidence: 0.35
  image_size: 1280
  device: cuda:0

ocr:
  base_url: http://127.0.0.1:8000
  timeout_seconds: 30
  max_retries: 2
  image_format: jpeg
  jpeg_quality: 95

output:
  runs_dir: runs
  save_crops: true
  save_raw_responses: true
  save_annotated_video: true
```

Endpoint yolu konfigurasyonda veya OCR adapter'inda NIM surumune gore sabitlenecektir. Uygulamanin geri kalani NVIDIA cevap semasina dogrudan baglanmayacaktir.

## 12. Riskler ve Onlemler

| Risk | Onlem |
| --- | --- |
| Nemotron genel OCR modelinin plakalarda yetersiz kalmasi | Etiketli baseline, hata siniflandirmasi ve sonraki fazda alternatif OCR karsilastirmasi |
| Kucuk veya hareketli plaka kirpimlari | Yuksek tespit cozunurlugu, kirpma payi ve sonraki fazda en iyi kare secimi |
| Her karede OCR nedeniyle dusuk hiz | Ilk olcumden sonra takip, tekrar engelleme ve batch islemi |
| NIM API semasinin surumle degismesi | Container surumunu sabitlemek ve adapter katmani kullanmak |
| Ag gecikmesi ve veri hacmi | Mumkunse tum pipeline'i GPU sunucusunda calistirmak; JPEG ve sinirli eszamanlilik |
| Yanlis format kurallarinin dogru OCR'ı bozmasi | Ham metni korumak, validasyon ile duzeltmeyi ayirmak |
| Plaka verisinin kisisel veri olmasi | Erisim kontrolu, saklama suresi, gereksiz goruntu kaydini kapatma ve KVKK ihtiyacini proje sahibinin degerlendirmesi |

## 13. Ilk Uygulama Sirasi

Kodlamaya baslandiginda izlenecek sira:

1. Proje iskeleti ve konfigurasyon.
2. Nemotron saglik kontrolu ve OCR istemcisi.
3. YOLO ile video tespiti ve plaka kirpimi.
4. Uctan uca pipeline.
5. JSONL/CSV ve isaretlenmis video.
6. Referans etiketleme formati ve metrik hesaplama.
7. Sabit video uzerinde baseline raporu.

Bu sirada ilk hedef maksimum FPS degil; hatalari kaybetmeyen, tekrar calistirilabilir ve iyilestirmelerin etkisini olcebilen bir sistemdir.
