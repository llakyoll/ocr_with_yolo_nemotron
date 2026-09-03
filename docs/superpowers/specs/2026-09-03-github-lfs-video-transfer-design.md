# Git LFS ile Video Aktarımı Tasarımı

## Amaç

Kaynak videoların iki veya daha fazla geliştirme bilgisayarı arasında GitHub
üzerinden aktarılabilmesini sağlamak.

## Kapsam

- Git LFS her geliştirme bilgisayarında bir kez etkinleştirilir.
- Kaynak video biçimleri (`.mp4`, `.mov`, `.avi`, `.mkv`) Git LFS tarafından
  izlenir.
- Videolar `data/input/` altında sürümlenir ve normal `git push` ile GitHub'a
  gönderilir.
- İkinci bilgisayarda `git clone` sonrasında `git lfs pull` gerçek video
  içeriğini indirir.

## Hariç Tutulanlar

- `runs/` altındaki üretilen videolar, OCR kırpımları ve raporlar sürümlenmez.
- Model ağırlıkları, gizli anahtarlar ve ortam dosyaları sürümlenmez.
- Büyük, sık değişen ham veri setleri için harici obje depolama bu ilk kurulumun
  dışında tutulur.

## Dosyalar ve Davranış

- `.gitattributes`: Video uzantılarını `filter=lfs` olarak kaydeder.
- `.gitignore`: Çalışma çıktıları, model dosyaları ve gizli yapılandırmaları
  hariç tutar; `data/input/` içindeki kaynak videoları hariç tutmaz.
- `README.md`: İlk bilgisayarda ekleme/push ve diğer bilgisayarda indirme
  komutlarını açıklar.
- `PLAN.md`: Veri yönetimi bölümüne Git LFS ve GitHub kota uyarısını ekler.

## Akış

İlk bilgisayarda geliştirici `git lfs install` çalıştırır, video dosyasını
`data/input/` altına koyar ve `git add`, `git commit`, `git push` kullanır.
Git LFS videoyu LFS deposuna yükler. Diğer bilgisayarda geliştirici Git LFS'yi
kurup etkinleştirir; depoyu klonladıktan sonra `git lfs pull` ile asıl video
dosyalarını indirir.

## Hata Yönetimi ve Doğrulama

- `git lfs ls-files` ilgili video dosyasını göstermelidir.
- GitHub LFS depolama ve bant genişliği kotası aşılırsa push veya indirme
  başarısız olabilir; bu durumda video dış depolamaya taşınır.
- Git LFS kurulu değilse ikinci bilgisayarda yalnızca küçük işaretçi dosyaları
  görülür; `git lfs install` ve `git lfs pull` çözümüdür.

## Başarı Kriteri

LFS ile commit edilen test videosu, temiz başka bir klonda `git lfs pull`
sonrasında gerçek dosya boyutu ve içeriğiyle kullanılabilir olur.
