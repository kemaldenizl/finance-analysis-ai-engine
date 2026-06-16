# Finansal Belge Analiz ve Yapay Zeka Motoru Raporu

## 1. Giriş

### 1.1 Projenin Genel Tanımı

Bu proje, finansal belge ve görsellerden işlem verisi çıkaran, bu veriyi standartlaştıran ve sonrasında yapay zeka destekli finansal analiz üreten bir backend sistemidir. Sistem; banka ekstresi, kredi kartı ekstresi, taranmış PDF, gerçek metin katmanına sahip PDF, mobil ekran görüntüsü ve kamera ile çekilmiş belge fotoğrafı gibi farklı girdi türlerini işleyebilecek şekilde tasarlanmıştır.

Proje FastAPI tabanlı bir API servisi olarak çalışır. Arka planda PostgreSQL veritabanı, Redis mesaj kuyruğu, Celery worker, lokal dosya saklama, OCR motoru, PDF işleme araçları, istatistiksel analiz servisleri, embedding tabanlı sınıflandırma ve opsiyonel Ollama/Qwen LLM entegrasyonu kullanır.

Temel amaç, kullanıcının finansal belgeyi sisteme yüklemesinden başlayarak şu çıktılara ulaşmaktır:

- Belgenin türünü otomatik belirlemek.
- Belgeyi veri çıkarımına uygun hale getirmek.
- İşlem satırlarını PDF text layer veya OCR ile çıkarmak.
- İşlemleri tarih, tutar, para birimi, açıklama, merchant, yön ve taksit bilgisiyle standart hale getirmek.
- Veri kalitesini puanlamak.
- Harcamaları kategorilere ayırmak.
- Harcama profili, anomali sinyalleri, gelecek dönem harcama tahmini ve taksit önerisi üretmek.
- Kullanıcı sorularına analiz bağlamında cevap verebilmek.

### 1.2 Problem Alanı

Finansal dokümanlar çoğu zaman farklı formatlarda gelir. Bazı PDF dosyalarında metin katmanı bulunur ve doğrudan okunabilir. Bazı PDF dosyaları sadece taranmış görüntülerden oluşur. Mobil bankacılık ekran görüntüleri, kamera ile çekilmiş fiş veya ekstre fotoğrafları ise OCR öncesi kalite iyileştirme gerektirir. Bu farklılıklar, tek bir veri çıkarım yaklaşımının yetersiz kalmasına neden olur.

Proje bu problemi aşamalı bir pipeline ile çözer:

| Problem | Projedeki Çözüm |
|---|---|
| Dosya türünün bilinmemesi | MIME tespiti ve kural tabanlı input sınıflandırma |
| PDF dosyalarının farklı yapıda olması | Real PDF, scanned PDF ve hybrid PDF ayrımı |
| Görsellerin OCR için kalitesiz olması | Görsel preprocessing, varyant üretimi ve OCR readiness skoru |
| OCR sonucunun hatalı veya gürültülü olması | Çoklu OCR varyant/PSM seçimi ve transaction parser filtreleri |
| Ham transaction verisinin standart olmaması | Normalization, merchant canonicalization ve validation |
| Finansal içgörü üretiminin zor olması | Kategori, profil, anomali, forecast ve taksit analiz servisleri |
| LLM servisinin her zaman erişilebilir olmaması | Deterministic fallback mimarisi |

Bu tablo, projenin yalnızca dosya yükleyen basit bir API olmadığını; çok biçimli finansal belge işleme problemini uçtan uca ele alan bir analiz motoru olduğunu gösterir.

### 1.3 Projenin Kapsamı

Proje beş ana işlevsel aşamadan oluşur:

| Aşama | Adı | Görevi | Ana Çıktı |
|---|---|---|---|
| Stage 1 | Input Classification | Yüklenen dosyanın türünü ve route bilgisini belirler | `ClassificationResult` |
| Stage 2 | Preprocessing | PDF/görselleri extraction için hazırlar | `PreprocessingResult` |
| Stage 3 | Data Extraction | İşlem satırlarını çıkarır | `PdfExtractionResult` |
| Stage 4 | Normalization | İşlemleri standart finansal kayıtlara dönüştürür | `Stage4Response` |
| Stage 5 | AI Analysis | Finansal analiz, özet ve sohbet cevabı üretir | `AnalyzeResponse` |

Bu aşamalar birbirinden ayrı servisler olarak modellenmiştir. Her aşama kendi sonucunu ayrı SQLAlchemy modeliyle veritabanına yazabilir. Böylece sistemde izlenebilirlik, hata ayrıştırma ve aşama bazlı tekrar çalıştırma kolaylaşır.

### 1.4 Projenin Hedef Kullanım Senaryosu

Sistem tipik olarak şu kullanım senaryosunu destekler:

1. Kullanıcı finansal PDF veya görsel dosyayı API’ye yükler.
2. Sistem dosyanın tipini ve işlenme rotasını belirler.
3. Celery worker belgeyi preprocessing aşamasından geçirir.
4. Kullanıcı veya üst seviye orchestrator uygun extraction endpointini çağırır.
5. Extraction sonucu normalize edilir.
6. Normalize edilmiş işlemler AI analizine gönderilir.
7. Sistem kategori dağılımı, anomali, harcama profili, harcama tahmini ve taksit önerisi üretir.
8. Kullanıcı isterse analiz hakkında soru sorar.

### 1.5 Temel Katkılar

Projenin öne çıkan katkıları şunlardır:

| Katkı | Açıklama |
|---|---|
| Çok formatlı belge kabulü | PDF, JPEG, PNG, WebP, HEIC ve HEIF dosyaları desteklenir |
| Belge tipi ayrımı | Real PDF, scanned PDF, hybrid PDF, screenshot, camera photo ayrımı yapılır |
| Akıllı preprocessing | OCR için birden fazla görüntü varyantı üretilir |
| Çoklu OCR seçimi | Her sayfa için farklı varyant ve PSM modları denenir |
| Ortak transaction parser | PDF ve OCR extraction aynı finansal satır parse mantığını paylaşır |
| Veri kalite skoru | Normalization sonrası satır ve özet güven skorları üretilir |
| AI analiz katmanı | Kategori, profil, anomali, forecast, taksit önerisi ve chat desteklenir |
| LLM bağımsız çalışma | LLM yoksa sistem deterministic fallback ile işlevini sürdürür |
| Docker ortamı | API, worker, PostgreSQL ve Redis compose ile ayağa kaldırılır |

## 2. Materyal ve Metod

### 2.1 Kullanılan Teknolojiler

Proje Python 3.12 ile geliştirilmiştir. Paket yönetiminde `uv`, web servis katmanında FastAPI kullanılmıştır.

| Katman | Kullanılan Teknoloji | Projedeki Rolü |
|---|---|---|
| API | FastAPI, Uvicorn | HTTP endpointleri ve servis sunumu |
| Veri doğrulama | Pydantic, Pydantic Settings | Request/response modelleri ve environment ayarları |
| Veritabanı | PostgreSQL, SQLAlchemy | Aşama sonuçlarının kalıcı olarak saklanması |
| Kuyruk | Redis, Celery | Preprocessing gibi arka plan işlerinin yürütülmesi |
| PDF işleme | PyMuPDF, pdfplumber | PDF sınıflandırma, text extraction ve render |
| Görsel işleme | OpenCV, Pillow, NumPy | Görsel kalite analizi ve preprocessing |
| OCR | Tesseract, pytesseract | Görsellerden text çıkarımı |
| Analitik | pandas, NumPy | Feature engineering ve finansal hesaplamalar |
| Anomali | PyOD ECOD, scikit-learn | İstatistiksel outlier tespiti |
| Embedding | sentence-transformers, torch | Merchant/category similarity sınıflandırması |
| LLM | Ollama, Qwen modeli | Özet, açıklama, kategori fallback ve sohbet |
| Container | Docker, Docker Compose | API, worker, DB ve Redis çalışma ortamı |

Bu teknoloji seçimi, sistemin hem klasik kural tabanlı veri çıkarımını hem de yapay zeka destekli analiz özelliklerini aynı backend içinde birleştirmesini sağlar.

### 2.2 Sistem Mimarisi

Uygulama `app/main.py` içinde oluşturulan FastAPI nesnesiyle başlar. Startup sırasında SQLAlchemy metadata kullanılarak tablolar oluşturulur. API routerları ayrı dosyalarda tutulur.

| Modül | Görev |
|---|---|
| `app/api/routes` | HTTP endpointleri |
| `app/core` | Konfigürasyon ve çekirdek ayarlar |
| `app/db` | SQLAlchemy engine ve session yönetimi |
| `app/models` | Veritabanı tabloları |
| `app/schemas` | Pydantic veri sözleşmeleri |
| `app/services/input` | Upload, MIME, classification ve routing |
| `app/services/preprocessing` | PDF render ve görüntü iyileştirme |
| `app/services/extraction` | PDF/OCR transaction extraction |
| `app/services/normalization` | Transaction normalize etme ve skor üretme |
| `app/services/ai` | Finansal AI analizi |
| `app/storage` | Lokal dosya saklama |
| `app/workers` | Celery task ve worker yapısı |

Mimari servis odaklıdır. Her servis tek bir ana sorumluluğa sahiptir. Bu ayrım, pipeline’ın farklı aşamalarını bağımsız test etmeyi ve geliştirmeyi kolaylaştırır.

### 2.3 Konfigürasyon Metodu

Proje ayarları `app/core/config.py` içinde `Settings` sınıfıyla yönetilir. `.env` dosyası desteklenir ve değerler environment variable olarak override edilebilir.

| Ayar Grubu | Örnek Değerler | Amaç |
|---|---|---|
| API | `APP_NAME`, `ENV`, `DEBUG`, `API_PORT` | Uygulama davranışı |
| Upload | `MAX_UPLOAD_SIZE_MB` | Dosya boyutu kontrolü |
| Storage | `LOCAL_INPUT_STORAGE_DIR`, `LOCAL_PROCESSED_STORAGE_DIR` | Lokal dosya yolları |
| DB/Queue | `DATABASE_URL`, `REDIS_URL` | PostgreSQL ve Redis bağlantıları |
| Preprocessing | `PDF_RENDER_DPI`, `PREPROCESSING_VERSION` | PDF render ve preprocessing ayarları |
| LLM | `LLM_ENABLED`, `LLM_BASE_URL`, `LLM_MODEL` | Ollama entegrasyonu |
| Embedding | `EMBEDDING_ENABLED`, `EMBEDDING_MODEL_NAME` | Semantic kategori sınıflandırması |
| Anomali | `ANOMALY_MIN_ROWS_FOR_PYOD`, cutoff değerleri | Outlier tespiti |
| Forecast | `FORECAST_LOOKBACK_MONTHS`, `FORECAST_TRAIN_EPOCHS` | Harcama tahmini |
| Kalite | `QUALITY_LOW_CONFIDENCE_THRESHOLD` | Analiz güven skoru |

Bu yapı, aynı kod tabanının lokal geliştirme, Docker ve farklı çalışma ortamlarında yapılandırılabilir olmasını sağlar.

### 2.4 Veri Saklama Metodu

Proje iki tür saklama kullanır:

1. Dosya saklama: Lokal filesystem.
2. Metadata ve sonuç saklama: PostgreSQL.

| Veri Türü | Saklama Yeri | Açıklama |
|---|---|---|
| Orijinal dosya | `/storage/inputs` | UUID tabanlı dosya adıyla saklanır |
| İşlenmiş görüntü varyantları | `/storage/processed/<input_id>` | OCR için üretilen PNG çıktılar |
| Upload metadata | `inputs` tablosu | Dosya adı, MIME, boyut, storage bilgisi |
| Classification sonucu | `input_classifications` tablosu | Kind, confidence, routing key |
| Preprocessing sonucu | `input_preprocessings` tablosu | Output, quality, readiness |
| Extraction sonucu | `data_extractions` tablosu | Transaction, debug, summary |
| Normalization sonucu | `normalizations` tablosu | Standart transaction ve skorlar |
| AI analiz sonucu | `ai_analysis_records` tablosu | Analiz çıktısı ve request snapshot |

Veritabanı tabloları migration sistemiyle değil, uygulama başlangıcında `Base.metadata.create_all()` ile oluşturulur.

### 2.5 Stage 1: Girdi Alma ve Sınıflandırma Metodu

Girdi alma işlemi `/v1/inputs` endpointi ile yapılır. Dosya multipart form ile alınır. Kullanıcı kimliği opsiyoneldir.

İlk validasyon adımları:

| Kontrol | Başarısız Durum |
|---|---|
| Dosya boş mu? | HTTP 400 |
| Dosya boyutu limiti aşıyor mu? | HTTP 413 |
| MIME tipi destekleniyor mu? | HTTP 415 |

Desteklenen MIME tipleri:

| MIME Tipi | Açıklama |
|---|---|
| `application/pdf` | PDF dosyaları |
| `image/jpeg` | JPEG görseller |
| `image/png` | PNG görseller |
| `image/webp` | WebP görseller |
| `image/heic` | HEIC görseller |
| `image/heif` | HEIF görseller |

Sınıflandırma sonucu şu türlerden biridir:

| Tür | Anlamı | Sonraki Rota |
|---|---|---|
| `real_pdf` | Metin katmanı güçlü PDF | Native PDF text extraction |
| `scanned_pdf` | Görsel ağırlıklı taranmış PDF | OCR |
| `hybrid_pdf` | Hem metin hem görsel içeren PDF | OCR / karma işleme |
| `screenshot` | Mobil/ekran görüntüsü | OCR |
| `camera_photo` | Kamera ile çekilmiş belge fotoğrafı | Preprocessing + OCR |
| `unknown` | Belirsiz dosya | Manual review |
| `unsupported` | Desteklenmeyen tür | İşlenmez |

PDF sınıflandırma PyMuPDF ile sayfa metni ve görsel alan oranları üzerinden yapılır. Görsel sınıflandırma EXIF verisi, blur skoru, kenar yoğunluğu, belge konturu, axis-aligned line score ve aspect ratio gibi OpenCV/Pillow özellikleriyle yapılır.

### 2.6 Stage 2: Ön İşleme Metodu

Ön işleme Celery worker tarafından `stage2.process_input` taskı ile yürütülür. Upload sonrası sistem bu taskı otomatik kuyruğa gönderir.

Kaynak türüne göre metot:

| Kaynak Türü | Ön İşleme Yaklaşımı | Extraction Metodu |
|---|---|---|
| `real_pdf` | Orijinal PDF korunur, render/OCR yapılmaz | `native_pdf_text` |
| `scanned_pdf` | Sayfalar görüntüye çevrilir, varyantlar üretilir | `ocr_multi_variant` |
| `hybrid_pdf` | Sayfalar görüntüye çevrilir, OCR varyantları üretilir | `ocr_multi_variant` |
| `screenshot` | Normalize ve grayscale OCR varyantları üretilir | `ocr_multi_variant` |
| `camera_photo` | Denoise, contrast, deskew, threshold varyantları üretilir | `ocr_multi_variant` |
| Diğer | İşlem atlanır, manual review | `manual_review` |

Üretilen varyant örnekleri:

| Kaynak | Varyant | Amaç |
|---|---|---|
| Screenshot | `normalized_original` | Renk/orijinal görünüm koruma |
| Screenshot | `ocr_grayscale` | Birincil OCR |
| Camera photo | `enhanced_grayscale` | OCR için iyileştirilmiş gri görüntü |
| Camera photo | `thresholded` | İkincil OCR denemesi |
| Scanned PDF | `rendered_original` | Fallback OCR |
| Scanned PDF | `enhanced_grayscale` | Birincil OCR |
| Scanned PDF | `thresholded` | İkincil OCR |

Her varyant için OCR readiness skoru hesaplanır. Skor; blur, parlaklık, kontrast, foreground density, connected component sayısı, component boyutu ve border artifact yoğunluğu gibi metriklere dayanır. En yüksek readiness değerine sahip tercih edilen varyant sonraki extraction aşamasında öncelik kazanır.

### 2.7 Stage 3: Veri Çıkarım Metodu

Veri çıkarım iki ana yolla yapılır:

| Extraction Türü | Kullanıldığı Girdi | Kullanılan Araç |
|---|---|---|
| Native PDF text extraction | `real_pdf` | pdfplumber, PyMuPDF |
| OCR multi-variant extraction | Screenshot, camera photo, scanned/hybrid PDF | Tesseract OCR |

Native PDF extraction önce `pdfplumber` ile text satırlarını çıkarır. Satır çıkarılamazsa PyMuPDF fallback olarak kullanılır.

OCR extraction’da her sayfa için birden fazla görüntü varyantı ve PSM modu denenir:

| PSM | Kullanım Amacı |
|---|---|
| 6 | Düzenli blok/satır yapıları |
| 11 | Seyrek text alanları |
| 4 | Kolon veya karma düzenler |

Her OCR sonucu şu sinyallerle skorlanır:

- Tarih bulunan satır sayısı.
- Para değeri bulunan satır sayısı.
- Hem tarih hem para içeren candidate satır sayısı.
- Ortalama OCR confidence.
- Toplam text uzunluğu.
- Varyant önceliği.
- Preprocessing’in preferred flag’i.

En yüksek final score’a sahip OCR sonucu sayfa için seçilir.

### 2.8 Transaction Parse Metodu

PDF ve OCR extraction aynı transaction parser mantığını kullanır. Parser finansal işlem satırlarını tarih, para, açıklama, taksit ve yön bilgisine göre çözer.

Desteklenen tarih formatları:

| Format | Örnek |
|---|---|
| Türkçe ay isimli | `5 Ocak 2025` |
| Gün/ay/yıl | `05.01.2025`, `05/01/2025` |
| ISO benzeri | `2025-01-05` |

Desteklenen para formatları:

| Format | Örnek |
|---|---|
| Para birimi önce | `₺ 450,00`, `USD 10.50` |
| Para birimi sonra | `450,00 TL`, `10.50 USD` |
| Binlik ayırıcı | `1.200,00`, `1,200.00` |
| ISO currency | `TRY`, `USD`, `EUR`, `GBP` ve pycountry destekli kodlar |

Taksit algılama örnekleri:

| Pattern | Anlamı |
|---|---|
| `1. Taksit` | Mevcut taksit |
| `(1/6)` | 6 taksitin 1.si |
| `100,00 x 6 = 600,00` | Birim, toplam taksit ve toplam tutar |

Parser ayrıca hesap özeti, kart limiti, asgari ödeme, ekstre numarası ve benzeri finansal gürültü satırlarını filtreler.

### 2.9 Stage 4: Normalizasyon Metodu

Normalizasyon aşamasında extraction’dan gelen ham transactionlar standart hale getirilir.

| Ham Alan | Normalize Alan | İşlem |
|---|---|---|
| `date` | `date` | ISO date formatına çevrilir |
| `price` veya `amount` | `amount` | Decimal ile 2 basamak yuvarlanır |
| `currency` | `currency` | Uppercase, `TL` -> `TRY` |
| `description` | `description` | Whitespace ve ayraç temizliği |
| Açıklama | `merchant` | MerchantNormalizer ile canonical form |
| `installment` | `installment` | Integer ve amount alanları normalize edilir |
| `confidence` | `confidence` | Skor servisi ile yeniden hesaplanır |

Validation uyarıları:

| Uyarı | Açıklama |
|---|---|
| `missing_or_invalid_date` | Tarih yok veya parse edilemiyor |
| `missing_amount` | Tutar yok |
| `non_positive_amount` | Tutar sıfır veya negatif |
| `missing_currency` | Para birimi yok |
| `invalid_currency` | Para birimi geçersiz |
| `missing_or_short_description` | Açıklama yok veya çok kısa |
| `invalid_installment_current_gt_total` | Mevcut taksit toplamdan büyük |

Duplicate temizleme; tarih, merchant, tutar, para birimi ve direction birleşimine göre yapılır. Aynı işlemden birden fazla kayıt varsa confidence değeri en yüksek olan tutulur.

### 2.10 Kalite Skorlama Metodu

Satır bazlı kalite skoru dört bileşenden oluşur:

| Bileşen | Ağırlık |
|---|---:|
| Extraction confidence | 0.40 |
| Field confidence | 0.25 |
| Completeness score | 0.20 |
| Validation score | 0.15 |

Field confidence; tarih, tutar, para birimi, açıklama ve merchant confidence değerlerinden oluşur. Completeness score zorunlu alanların doluluk oranını ölçer. Validation score ise valid/warning/invalid durumuna göre hesaplanır.

Bu sayede sistem sadece transaction çıkarmakla kalmaz; her satırın ne kadar güvenilir olduğunu da nicel olarak ifade eder.

### 2.11 Stage 5: Yapay Zeka Analiz Metodu

AI analysis katmanı normalize edilmiş transactionlar üzerinden çalışır. Ana servis `AIAnalysisService`tir.

| Alt Servis | Görevi |
|---|---|
| `FeatureEngineeringService` | Transactionları pandas dataframe’e dönüştürür |
| `CategorizationService` | Harcamaları kategorilere ayırır |
| `SpendingProfileService` | Harcama davranışı etiketleri üretir |
| `AnomalyDetectionService` | Olağan dışı işlem sinyallerini bulur |
| `ForecastInstallmentService` | Harcama tahmini ve taksit önerisi üretir |
| `LLMReportService` | Özet, açıklama ve soru cevabı üretir |
| `OllamaProvider` | LLM erişimini sağlar |
| `EmbeddingCategoryClassifier` | Semantic kategori eşleşmesi yapar |

Feature engineering sonucunda tarih, ay, hafta sonu bilgisi, harcama tutarı, kredi tutarı, signed amount, log amount, taksit varlığı, yabancı para kullanımı, düşük confidence ve invalid flag gibi kolonlar oluşturulur.

### 2.12 Kategori Sınıflandırma Metodu

Kategori sınıflandırma çok katmanlıdır:

| Sıra | Yöntem | Açıklama |
|---|---|---|
| 1 | Direction rule | Credit işlemler `income_or_refund` olur |
| 2 | Regex/rule matching | YAML taksonomi patternleri kullanılır |
| 3 | Embedding similarity | Merchant text semantic olarak taksonomiye eşlenir |
| 4 | LLM fallback | Kalan işlemler Ollama/Qwen ile sınıflandırılır |
| 5 | Other fallback | Emin olunmayan işlem `other` olur |

Taksonomi kategorileri:

| Kategori ID | Türkçe Etiket |
|---|---|
| `travel` | Seyahat |
| `food` | Yeme İçme |
| `groceries` | Market |
| `fuel` | Akaryakıt |
| `transport` | Ulaşım |
| `health` | Sağlık |
| `clothing` | Giyim |
| `entertainment` | Eğlence |
| `telecom` | İletişim |
| `subscription` | Abonelik |
| `shopping` | Alışveriş |
| `accommodation` | Konaklama |
| `education` | Eğitim |
| `insurance` | Sigorta |
| `utilities_tax` | Fatura ve Vergi |
| `finance` | Bankacılık |
| `payment` | Kart Ödemesi |
| `other` | Diğer |

### 2.13 Anomali Tespit Metodu

Anomali tespiti debit transactionlar üzerinde yapılır.

| Veri Durumu | Kullanılan Metot |
|---|---|
| Yeterli geçmiş veri varsa | PyOD ECOD |
| Yeterli geçmiş veri yoksa | Robust statistical fallback |

PyOD ECOD için kullanılan feature set:

- `log_amount`
- `has_installment`
- `is_foreign_currency`
- `is_low_confidence`
- `is_weekend`

Business flag bonusları:

| Flag | Bonus |
|---|---:|
| `foreign_currency_transaction` | 0.12 |
| `installment_transaction` | 0.05 |
| `low_source_confidence` | 0.18 |
| `source_validation_warning` | 0.18 |

Severity sınıfları:

| Skor Aralığı | Severity |
|---|---|
| `>= 0.70` | High |
| `>= 0.45` | Medium |
| `< 0.45` | Low |

### 2.14 Forecast ve Taksit Önerisi Metodu

Forecast servisi aylık debit harcama serisi üzerinden gelecek dönem harcamasını tahmin eder.

| Veri Miktarı | Metot |
|---|---|
| Veri yok | `no_data` |
| Transformer için yetersiz ay | Weighted moving average fallback |
| Yeterli ay | Transformer + moving average ensemble |

Transformer modeli küçük bir `MonthlySpendTransformer` sınıfıdır. Model request sırasında eldeki aylık seriyle eğitilir. Daha az veri olduğunda ağırlıklı hareketli ortalama kullanılır.

Taksit önerisi için:

- Kullanıcı `purchase_scenario` gönderir.
- Tutar ve maksimum taksit sayısı alınır.
- Her ay seçeneği için aylık ödeme hesaplanır.
- Bu aylık ödeme forecast edilen aylık harcamaya bölünür.
- Burden ratio düşük olan ilk seçenek önerilir.

Risk hesabı:

| Monthly Burden Ratio | Risk |
|---|---|
| `<= 0.15` | Low |
| `<= 0.30` | Medium |
| `> 0.30` | High |

Bu çıktı kredi tavsiyesi değildir; sistem bunu spending burden estimate olarak uyarı listesine ekler.

### 2.15 LLM Kullanım Metodu

LLM entegrasyonu Ollama üzerinden yapılır. Varsayılan model `qwen3:8b` olarak ayarlanmıştır.

LLM’in kullanıldığı yerler:

| Kullanım | Görev |
|---|---|
| Kategori fallback | Rule/embedding ile çözülemeyen merchantları sınıflandırmak |
| Anomali açıklaması | Tespit edilen anomaly itemlarını kullanıcı dilinde özetlemek |
| Taksit açıklaması | Hesaplanmış taksit önerisini anlaşılır metne çevirmek |
| Executive summary | Analiz bulgularını kısa özet haline getirmek |
| Chat | Kullanıcı sorularına analiz bağlamında cevap vermek |

LLM kapalıysa veya Ollama erişilemezse sistem deterministic template cevapları kullanır. Bu özellik, projenin temel işlevlerinin LLM bağımlılığı olmadan devam etmesini sağlar.

### 2.16 API Metodu ve Endpointler

| Endpoint | Metot | Amaç |
|---|---|---|
| `/health` | GET | Servis, PostgreSQL ve Redis durumu |
| `/health/live` | GET | Uygulamanın canlılık kontrolü |
| `/health/ready` | GET | Hazır olma kontrolü |
| `/v1/inputs` | POST | Dosya yükleme ve sınıflandırma |
| `/v1/preprocessings/{input_id}` | GET | Son preprocessing sonucunu okuma |
| `/v1/extractions/pdf/{input_id}` | POST | Real PDF extraction |
| `/v1/extractions/image/{input_id}` | POST | OCR extraction |
| `/v1/extractions/{input_id}/latest` | GET | Son extraction sonucunu okuma |
| `/v1/normalizations/{input_id}` | POST | Son extraction sonucunu normalize etme |
| `/v1/normalizations/{input_id}/latest` | GET | Son normalization sonucunu okuma |
| `/v1/ai/analyze` | POST | AI analysis üretme |
| `/v1/ai/analyze-and-save` | POST | AI analysis üretip DB’ye kaydetme |
| `/v1/ai/analyses/{input_id}/latest` | GET | Son AI analizini okuma |
| `/v1/ai/chat` | POST | Analiz bağlamında soru cevaplama |
| `/v1/ai/health` | GET | AI servis durumunu okuma |

### 2.17 Test Metodu

Projede pytest tabanlı test yapısı bulunur.

| Test Dosyası | Test Edilen Alan |
|---|---|
| `tests/test_health.py` | `/health/live` endpointi |
| `tests/test_analyze_contract.py` | AI analysis response contractı ve kategori kuralları |
| `tests/conftest.py` | Test storage ayarları |

Test fixture dosyaları:

| Fixture | Amaç |
|---|---|
| `real_pdf.pdf` | Gerçek PDF senaryosu |
| `scanned_pdf.pdf` | Taranmış PDF senaryosu |
| `screenshot.jpg` | Ekran görüntüsü senaryosu |
| `camera_photo.jpeg` | Kamera fotoğrafı senaryosu |

Mevcut testler özellikle health check ve AI analysis contractını doğrular. Fixture dosyaları bulunmasına rağmen upload, preprocessing, OCR extraction, native PDF extraction ve normalization için geniş uçtan uca test kapsamı mevcut kodda sınırlıdır.

## 3. Bulgular

### 3.1 Genel Sistem Bulguları

Proje incelendiğinde, sistemin tek parça bir OCR servisi değil; çok aşamalı, izlenebilir ve genişletilebilir bir finansal belge analiz pipeline’ı olduğu görülmüştür.

| İncelenen Alan | Bulgu |
|---|---|
| Mimari | API, worker, storage, DB ve AI servisleri ayrılmıştır |
| Veri akışı | Her aşama kendi çıktı modeline ve DB kaydına sahiptir |
| Input desteği | PDF ve yaygın görsel formatları desteklenmektedir |
| Extraction yaklaşımı | Real PDF ve OCR yolları ayrılmıştır |
| Veri kalitesi | Normalization sonrası confidence ve validation skorları üretilir |
| AI yaklaşımı | Deterministic, embedding, istatistiksel ve LLM yöntemleri birlikte kullanılır |
| Çalışma ortamı | Docker Compose ile API, worker, PostgreSQL ve Redis birlikte ayağa kalkar |

### 3.2 Sınıflandırma Bulguları

Sınıflandırma katmanı dosya türünü belirlemede kural tabanlı ve açıklanabilir bir yaklaşım kullanır.

PDF tarafında:

- Metin yoğunluğu yüksek ve text page ratio güçlü dosyalar `real_pdf` olarak ayrılır.
- Görsel alan oranı yüksek, metni zayıf PDF’ler `scanned_pdf` olarak ayrılır.
- Hem metin hem görsel içeren dosyalar `hybrid_pdf` olarak işaretlenir.

Görsel tarafında:

- EXIF, blur, edge density, belge konturu ve aspect ratio birlikte kullanılır.
- Screenshot ve camera photo ayrımı tek bir sinyale bağlı değildir.
- Düşük güvenli durumlarda warning üretilir.

Bu yaklaşım, kullanıcının dosya türünü manuel belirtmesine ihtiyaç bırakmadan sonraki aşamanın seçilmesini sağlar.

### 3.3 Preprocessing Bulguları

Preprocessing aşamasında sistemin temel gücü tek bir görüntü çıktısı üretmek yerine OCR için çoklu aday üretmesidir.

| Özellik | Bulgu |
|---|---|
| Real PDF işleme | Orijinal PDF korunur, gereksiz OCR yapılmaz |
| Scanned PDF işleme | Sayfalar yüksek DPI ile görüntüye çevrilir |
| Screenshot işleme | Hafif grayscale/contrast varyantı üretilir |
| Camera photo işleme | Denoise, contrast, deskew ve threshold uygulanır |
| Varyant seçimi | OCR readiness score’a göre yapılır |
| Storage | Her varyant processed storage altında saklanır |

Bu yapı, OCR başarısını artırmaya yöneliktir. Özellikle farklı belge kalitelerinde tek bir preprocessing stratejisine bağlı kalmamak önemli bir avantajdır.

### 3.4 Extraction Bulguları

Extraction aşaması iki farklı kaynak tipine göre ayrılır:

| Kaynak | Güçlü Yan | Risk |
|---|---|---|
| Real PDF | Text layer doğrudan okunur, OCR hatası yoktur | PDF layout satırları bölünebilir |
| OCR | Screenshot ve scanned belge desteği sağlar | OCR karakter hataları ve satır bölünmeleri olabilir |

Parser bu riskleri azaltmak için:

- Tarih ve para desenlerini birlikte arar.
- Tarih var fakat para yoksa yakın satırları merge eder.
- Taksit bilgilerini ayrıca yakalar.
- Para birimini doküman genelinden tahmin eder.
- Gürültü satırlarını filtreler.
- Confidence skoru üretir.

OCR tarafında varyant seçimi, en fazla transaction sinyali veren OCR sonucunu seçerek başarımı artırmaya çalışır.

### 3.5 Normalizasyon Bulguları

Normalizasyon aşaması, extraction sonucunu analiz edilebilir veriye dönüştüren kritik katmandır.

| Normalizasyon Yeteneği | Bulgu |
|---|---|
| Tarih standardizasyonu | ISO date formatı hedeflenir |
| Tutar standardizasyonu | İki ondalık basamağa yuvarlama yapılır |
| Para birimi | ISO kod mantığına yakın normalize edilir |
| Merchant | Alias ve canonical form desteği vardır |
| Duplicate temizleme | Aynı işlemden güveni yüksek olan tutulur |
| Validation | Eksik/hatalı alanlar warning veya invalid olur |
| Confidence | Satır skoru çok bileşenli hesaplanır |

Bu aşama olmadan AI analiz katmanının güvenilir sonuç üretmesi zorlaşır. Proje bu nedenle extraction ve analysis arasına güçlü bir veri kalite katmanı yerleştirmiştir.

### 3.6 AI Analiz Bulguları

AI analysis katmanı yalnızca LLM çağıran basit bir servis değildir. Kural tabanlı, embedding tabanlı, istatistiksel ve LLM destekli yaklaşımları birlikte kullanır.

| Analiz Özelliği | Üretilen Sonuç |
|---|---|
| Kategorilendirme | Transaction bazlı kategori ve kategori özeti |
| Harcama profili | Profil label’ları, yoğun kategori, oranlar |
| Anomali tespiti | Riskli/olağan dışı işlem sinyalleri |
| Forecast | Gelecek dönem tahmini harcama |
| Taksit önerisi | Senaryo bazlı ay ve risk önerisi |
| Executive summary | Finansal durumun kısa özeti |
| Chat | Analiz bağlamında soru-cevap |

Analiz güveni, source confidence değerinden düşük güvenli ve invalid satır cezaları düşülerek hesaplanır. Bu, hatalı veya eksik extraction sonucunun AI analiz statüsüne yansımasını sağlar.

### 3.7 LLM ve Deterministic Fallback Bulguları

Proje LLM’i ana karar motoru olarak değil, destekleyici açıklama ve fallback katmanı olarak kullanır. Bu tasarım önemli bir güvenilirlik sağlar.

| Durum | Sistem Davranışı |
|---|---|
| LLM aktif ve erişilebilir | Özet, açıklama, chat ve fallback kategoriler LLM ile desteklenir |
| LLM kapalı | Deterministic template cevapları kullanılır |
| LLM erişilemez | Warning üretilir ve fallback kullanılır |
| Structured LLM cevabı şemaya uymazsa | Yeniden denenir, olmazsa `None` döner |

Bu bulgu, sistemin üretim ortamında LLM bağımlılığı yüzünden tamamen durmayacağını gösterir.

### 3.8 Veritabanı ve İzlenebilirlik Bulguları

Her aşama ayrı tabloya yazıldığı için pipeline izlenebilir durumdadır.

| Tablo | İzlenen Aşama |
|---|---|
| `inputs` | Upload metadata |
| `input_classifications` | Stage 1 |
| `input_preprocessings` | Stage 2 |
| `data_extractions` | Stage 3 |
| `normalizations` | Stage 4 |
| `ai_analysis_records` | Stage 5 |

Bu yapı sayesinde bir input için hangi aşamada hangi kararın verildiği, hangi outputun üretildiği ve hangi confidence skorlarının oluştuğu sonradan incelenebilir.

### 3.9 API ve Operasyon Bulguları

API tasarımı aşama bazlıdır. Upload sonrası preprocessing otomatik Celery’ye dispatch edilir; extraction, normalization ve AI analysis endpointleri ayrıca çağrılır.

Bu yapı iki farklı kullanım biçimine olanak tanır:

- Manuel/API kontrollü pipeline yürütme.
- Üst seviye bir orchestration servisiyle aşamaları sırayla çağırma.

Health endpointleri PostgreSQL ve Redis bağımlılıklarının durumunu kontrol eder. Docker Compose dosyalarında API ve worker bu servislerin healthy olmasını bekler.

### 3.10 Test Bulguları

Mevcut testler temel sözleşmeler için faydalıdır:

- Health endpointinin çalıştığı doğrulanır.
- AI analysis response modelinin serialize/validate döngüsü test edilir.
- Bilinen merchantların kategori kurallarıyla eşleştiği doğrulanır.

Ancak kapsam bakımından:

| Eksik veya Sınırlı Test Alanı | Açıklama |
|---|---|
| Upload endpointi | Multipart dosya yükleme uçtan uca test edilmiyor |
| PDF classification | Fixture olmasına rağmen ayrı test kapsamı görünmüyor |
| Image classification | Screenshot/camera ayrımı test edilmiyor |
| Preprocessing | Varyant üretimi ve readiness skoru test edilmiyor |
| OCR extraction | Tesseract tabanlı extraction test edilmiyor |
| Normalization | Duplicate, validation ve skor testleri geniş değil |
| Celery worker | Worker task akışı test edilmiyor |

Bu durum projenin çalışmadığı anlamına gelmez; sadece mevcut otomatik test güvence alanının daha çok AI contract ve health check ile sınırlı olduğunu gösterir.

### 3.11 Güçlü Yönler

| Güçlü Yön | Açıklama |
|---|---|
| Aşamalı pipeline | Her görev ayrılmış ve izlenebilir |
| Çok format desteği | PDF ve görsel kaynaklar desteklenir |
| OCR öncesi varyant üretimi | Farklı kalite koşullarına uyum sağlar |
| Açıklanabilir parser | Regex ve kural tabanlı transaction çıkarımı anlaşılırdır |
| Veri kalite skoru | Analiz güvenilirliği ölçülebilir hale gelir |
| LLM fallback tasarımı | LLM yokken sistem temel işlevlerini kaybetmez |
| Taksonomi tabanlı kategori | YAML ile genişletilebilir kategori sistemi vardır |
| Dockerlaştırma | Yerel çalıştırma ve bağımlılık yönetimi kolaylaşır |

### 3.12 Sınırlılıklar

| Sınırlılık | Etki |
|---|---|
| Authentication yok | API güvenliği dış katmana bırakılmıştır |
| Migration sistemi yok | Şema değişiklikleri üretim ortamında dikkat ister |
| Lokal storage | Dağıtık/ölçekli dosya saklama için S3 benzeri çözüm gerekebilir |
| OCR bağımlılığı | Tesseract dil paketi ve görüntü kalitesi sonucu etkiler |
| LLM/Ollama bağımlılığı | LLM özellikleri servis erişilebilirliğine bağlıdır |
| Test kapsamı sınırlı | Uçtan uca kalite güvencesi genişletilmelidir |
| Pipeline tam otomatik değil | Extraction ve normalization ayrı endpoint çağrısı gerektirir |

## 4. Sonuç

### 4.1 Genel Değerlendirme

Bu proje, finansal belge analizi için kapsamlı ve modüler bir backend altyapısı sunmaktadır. Sistem yalnızca OCR yapan veya yalnızca LLM ile yorum üreten bir yapı değildir. Bunun yerine dosya kabulünden başlayıp sınıflandırma, preprocessing, extraction, normalization, kalite skoru ve AI analizine kadar uzanan bütünlüklü bir pipeline kurar.

Mimari olarak en önemli başarı, farklı belge tiplerinin tek bir işlem akışına zorlanmaması; her belge türü için uygun yöntemin seçilmesidir. Real PDF’lerde text layer kullanılarak gereksiz OCR maliyetinden kaçınılır. Görsel ve scanned dokümanlarda ise OCR öncesi görüntü varyantları üretilerek veri çıkarım başarısı artırılmaya çalışılır.

### 4.2 Projenin Yaptığı İşlerin Özeti

| Alan | Projenin Yaptığı İş |
|---|---|
| Dosya kabulü | PDF ve görsel dosyaları alır, validasyon yapar |
| Sınıflandırma | Dosyayı real/scanned/hybrid PDF veya screenshot/camera photo olarak ayırır |
| Preprocessing | OCR’a uygun görüntü varyantları üretir |
| Extraction | Native PDF text veya Tesseract OCR ile transaction çıkarır |
| Parse | Tarih, tutar, para birimi, taksit, direction ve açıklama çözer |
| Normalization | Transactionları standart finansal kayda çevirir |
| Skorlama | Satır ve genel confidence skorları üretir |
| Kategorilendirme | Rule, embedding ve LLM fallback ile kategori belirler |
| Profil | Harcama davranışı label’ları üretir |
| Anomali | İstatistiksel ve iş kuralı destekli riskli işlem sinyalleri bulur |
| Forecast | Gelecek dönem harcama tahmini üretir |
| Taksit | Satın alma senaryosu için risk bazlı taksit önerir |
| Chat | Analiz bağlamındaki sorulara cevap verir |
| Kalıcılık | Her aşama sonucunu PostgreSQL’de saklar |
| Operasyon | Docker, Redis, Celery ve healthcheck desteği sağlar |

### 4.3 Akademik ve Teknik Katkı

Akademik açıdan proje, finansal belge işleme probleminde çok aşamalı bir yaklaşım sunmaktadır. Materyal olarak PDF/görsel belgeler; metod olarak kural tabanlı sınıflandırma, görüntü işleme, OCR, regex tabanlı bilgi çıkarımı, veri normalizasyonu, istatistiksel anomali tespiti, embedding tabanlı kategori sınıflandırması ve LLM destekli doğal dil açıklama yöntemleri birlikte kullanılmıştır.

Teknik açıdan proje şu özellikleriyle değerlidir:

- Veri çıkarım ve analiz katmanlarını birbirinden ayırır.
- Her aşamada confidence ve warning bilgisi üretir.
- AI sonucunun kalitesini kaynak veri kalitesiyle ilişkilendirir.
- LLM’i kontrolsüz ana karar verici değil, sınırlı ve bağlamlı yardımcı olarak konumlandırır.
- Genişletilebilir taksonomi ve servis yapısı sunar.

### 4.4 Geliştirme Önerileri

Proje mevcut haliyle işlevsel bir çekirdek sunmaktadır. Daha ileri kullanım için öneriler:

| Öneri | Beklenen Katkı |
|---|---|
| Authentication/API key eklenmesi | Servis güvenliğini artırır |
| Alembic migration kurulması | Veritabanı şema yönetimini güçlendirir |
| Uçtan uca pipeline endpointi | Kullanıcı tarafındaki orchestration yükünü azaltır |
| S3/MinIO storage desteği | Dağıtık deployment için uygunluk sağlar |
| Geniş test kapsamı | OCR, normalization ve worker güvenilirliğini artırır |
| Banka bazlı parser profilleri | Farklı ekstre formatlarında başarıyı artırır |
| Monitoring ve structured logging | Üretim ortamı izlenebilirliğini artırır |
| Queue retry politikaları | Worker hata toleransını iyileştirir |

### 4.5 Nihai Sonuç

Sonuç olarak proje, finansal belge analizini uçtan uca ele alan, modern Python backend teknolojileriyle geliştirilmiş, yapay zeka destekli ve modüler bir analiz motorudur. Sistemin en güçlü tarafı, karmaşık finansal belge problemini tek seferlik bir OCR işlemi olarak değil, aşama aşama doğrulanan ve zenginleştirilen bir veri işleme süreci olarak tasarlamasıdır.

Bu yapı sayesinde sistem; belgeleri tanıyabilir, uygun preprocessing stratejisini seçebilir, transactionları çıkarabilir, veriyi normalize edebilir, kalite skorlarıyla güvenilirliği ölçebilir ve kullanıcıya kategori, profil, anomali, tahmin, taksit ve sohbet çıktıları sağlayabilir. Geliştirme alanları bulunsa da mevcut mimari, bitirme projesi kapsamında güçlü, açıklanabilir ve genişletilebilir bir finansal analiz altyapısı ortaya koymaktadır.
