# Finance Analysis AI Engine Proje Dokümantasyonu

## 1. Projenin Amacı

Bu proje, finansal dokümanlardan işlem verisi çıkarmak, bu veriyi normalize etmek ve ardından yapay zeka destekli finansal analiz üretmek için geliştirilmiş FastAPI tabanlı bir backend servisidir. Sistem banka ekstresi, kredi kartı ekstresi, finansal işlem ekran görüntüsü veya kamera ile çekilmiş belge fotoğrafı gibi farklı dosya tiplerini kabul eder.

Temel hedefler şunlardır:

- PDF veya görsel dosyanın türünü otomatik sınıflandırmak.
- Real PDF, scanned PDF, hybrid PDF, screenshot ve camera photo ayrımını yapmak.
- Dosyayı sonraki aşamaya uygun hale getirmek için ön işleme uygulamak.
- PDF text layer veya OCR üzerinden işlem satırlarını çıkarmak.
- Ham işlem satırlarını standart ve skorlanabilir finansal kayıtlara dönüştürmek.
- Harcama kategorisi, profil, anomali, gelecek dönem harcama tahmini ve taksit önerisi üretmek.
- İsteğe bağlı olarak Ollama/Qwen tabanlı LLM ile açıklama, özet ve sohbet cevapları üretmek.
- Her aşamanın sonucunu PostgreSQL üzerinde saklamak.
- Ön işleme gibi arka plan işlerini Redis/Celery ile yürütmek.

README dosyasında proje “FastAPI based finance analysis ai engine” olarak tanımlanmıştır. Mevcut sürüm notu `v0.2.2 Input Preprocessing Advanced` olarak geçer ve PDF sayfa ilgililik tespiti, image/PDF finans bölgesi tespiti gibi Stage 2 geliştirmelerine işaret eder. Kod tabanında aktif olarak görülen mimari beş ana aşamadan oluşur: input classification, preprocessing, extraction, normalization ve AI analysis.

## 2. Teknoloji Yığını

Proje Python 3.12 ile geliştirilmiştir. Paket yönetimi için `uv` kullanılır.

Ana bağımlılıklar:

- `fastapi`: HTTP API sunucusu.
- `uvicorn[standard]`: FastAPI uygulamasını çalıştırmak için ASGI sunucusu.
- `python-multipart`: Dosya upload desteği.
- `pydantic`, `pydantic-settings`: Veri doğrulama ve environment ayarları.
- `sqlalchemy`, `psycopg2-binary`: PostgreSQL bağlantısı ve ORM.
- `redis`, `celery`: Arka plan iş kuyruğu.
- `python-magic`: MIME type tespiti.
- `pymupdf`, `pdfplumber`: PDF okuma, text extraction ve render işlemleri.
- `pillow`, `opencv-python-headless`, `numpy`: Görsel işleme.
- `pytesseract`: OCR.
- `pandas`: Analiz ve feature engineering.
- `pycountry`: Para birimi kodları.
- `httpx`: Ollama HTTP entegrasyonu.
- `pyod`, `scikit-learn`: Anomali tespiti.
- `sentence-transformers`, `torch`: Embedding sınıflandırması ve Transformer tabanlı tahmin.
- `pyyaml`: Kategori taksonomisini YAML dosyasından okumak.
- `python-dotenv`: `.env` desteği.

Geliştirme bağımlılıkları:

- `pytest`, `pytest-asyncio`: Testler.
- `httpx`: TestClient ve HTTP testleri.
- `ruff`: Lint.
- `mypy`: Tip kontrolü.

## 3. Proje Dizin Yapısı

Kök dizindeki önemli dosyalar:

- `README.md`: Kısa proje tanımı, özellik listesi ve lokal geliştirme komutları.
- `pyproject.toml`: Paket metadata, bağımlılıklar, ruff/mypy/pytest ayarları.
- `uv.lock`: Kilitlenmiş dependency seti.
- `Dockerfile`: API/worker container imajı.
- `docker-compose.yml`: Üretime yakın yerel compose kurulumu.
- `docker-compose.dev.yml`: Hot reload ve lokal volume mount içeren geliştirme compose kurulumu.
- `project_documentation.md`: Bu dokümantasyon dosyası.
- `tests/`: Testler ve fixture dosyaları.
- `app/`: Ana uygulama kodu.

`app/` altındaki ana modüller:

- `app/main.py`: FastAPI uygulamasını oluşturur ve routerları bağlar.
- `app/api/routes/`: HTTP endpointleri.
- `app/core/`: Konfigürasyon ve çekirdek yardımcılar.
- `app/db/`: SQLAlchemy engine/session ve tablo oluşturma.
- `app/models/`: SQLAlchemy veri modelleri.
- `app/schemas/`: Pydantic request/response şemaları.
- `app/services/`: İş mantığı servisleri.
- `app/storage/`: Lokal dosya saklama yardımcıları.
- `app/workers/`: Celery uygulaması ve tasklar.

`app/services/` altındaki ana domain ayrımı:

- `input/`: Dosya yükleme, MIME tespiti, PDF/görsel sınıflandırma ve routing.
- `preprocessing/`: PDF render, görsel iyileştirme, OCR readiness analizi ve varyant üretimi.
- `extraction/`: Native PDF text extraction, OCR extraction, transaction parser.
- `normalization/`: Transaction normalize etme, merchant normalize etme, kalite skorları.
- `ai/`: Kategorilendirme, embedding, LLM, anomali, profil, tahmin ve taksit önerisi.

## 4. Uygulama Başlangıcı

Ana uygulama `app/main.py` içinde tanımlanır.

`create_app()` fonksiyonu:

- `FastAPI` nesnesini oluşturur.
- Uygulama adını `settings.APP_NAME` üzerinden alır.
- Versiyonu `0.1.0` olarak verir.
- Debug modunu `settings.DEBUG` üzerinden belirler.
- Aşağıdaki routerları bağlar:
  - health
  - inputs
  - preprocessings
  - extractions
  - normalizations
  - ai-analysis
- Startup event içinde `create_db_tables()` çağırır.

`create_db_tables()` SQLAlchemy metadata üzerinden tabloları doğrudan oluşturur. Projede Alembic migration yapısı bulunmaz; tablo oluşturma uygulama açılışında yapılır.

## 5. Konfigürasyon

Konfigürasyon `app/core/config.py` içindeki `Settings` sınıfıyla yönetilir. `BaseSettings` kullanıldığı için değerler `.env` dosyasından ve environment variable değerlerinden okunabilir. `get_settings()` fonksiyonu `lru_cache` ile cache’lenir ve modül sonunda `settings` singleton’ı oluşturulur.

Önemli ayarlar:

- `APP_NAME`: Varsayılan `Finance AI Input Pipeline`.
- `ENV`: Varsayılan `dev`.
- `DEBUG`: Varsayılan `True`.
- `API_HOST`: Varsayılan `0.0.0.0`.
- `API_PORT`: Varsayılan `8000`.
- `MAX_UPLOAD_SIZE_MB`: Varsayılan `50`.
- `DATABASE_URL`: Varsayılan PostgreSQL bağlantısı `postgresql+psycopg2://postgres:postgres@postgres:5432/finance_ai`.
- `REDIS_URL`: Varsayılan Redis bağlantısı `redis://redis:6379/0`.
- `LOCAL_STORAGE_ROOT`: Varsayılan `/storage`.
- `LOCAL_INPUT_STORAGE_DIR`: Varsayılan `/storage/inputs`.
- `LOCAL_PROCESSED_STORAGE_DIR`: Varsayılan `/storage/processed`.
- `CLASSIFICATION_MODEL_VERSION`: Varsayılan `rules-v1`.
- `PREPROCESSING_VERSION`: Varsayılan `preprocessing-v2-no-crop`.
- `PDF_RENDER_DPI`: Varsayılan `220`.
- `PREPROCESSING_SAVE_DEBUG_VARIANTS`: Varsayılan `True`.
- `PREPROCESSING_MAX_OUTPUT_VARIANTS_PER_PAGE`: Varsayılan `4`.
- `AI_ANALYSIS_VERSION`: Varsayılan `ai-analysis-v2`.

LLM ayarları:

- `LLM_ENABLED`: Varsayılan `True`.
- `LLM_BASE_URL`: Varsayılan `http://ollama:11434`.
- `LLM_MODEL`: Varsayılan `qwen3:8b`.
- `LLM_TIMEOUT_SECONDS`: Varsayılan `180`.
- `LLM_TEMPERATURE`: Varsayılan `0.1`.
- `LLM_NUM_CTX`: Varsayılan `4096`.
- `LLM_SEED`: Varsayılan `42`.
- `LLM_TOP_P`: Varsayılan `0.9`.
- `LLM_TOP_K`: Varsayılan `40`.
- `LLM_REPEAT_PENALTY`: Varsayılan `1.1`.
- `LLM_NUM_PREDICT`: Varsayılan `256`.
- `LLM_MAX_RETRIES`: Varsayılan `1`.
- `LLM_KEEP_ALIVE`: Varsayılan `5m`.
- `LLM_CHAT_NUM_PREDICT`: Varsayılan `256`.

Embedding ayarları:

- `EMBEDDING_ENABLED`: Varsayılan `True`.
- `EMBEDDING_MODEL_NAME`: Varsayılan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `EMBEDDING_SIMILARITY_THRESHOLD`: Varsayılan `0.52`.

Anomali ayarları:

- `ANOMALY_MIN_ROWS_FOR_PYOD`: Varsayılan `8`.
- `ANOMALY_CONTAMINATION`: Varsayılan `0.12`.
- `ANOMALY_PYOD_SCORE_CUTOFF`: Varsayılan `0.40`.
- `ANOMALY_ROBUST_SCORE_CUTOFF`: Varsayılan `0.25`.

Forecast ayarları:

- `FORECAST_MIN_MONTHS_TRANSFORMER`: Varsayılan `6`.
- `FORECAST_LOOKBACK_MONTHS`: Varsayılan `3`.
- `FORECAST_TRAIN_EPOCHS`: Varsayılan `120`.
- `FORECAST_RANDOM_SEED`: Varsayılan `42`.

Kalite ayarları:

- `QUALITY_LOW_CONFIDENCE_THRESHOLD`: Varsayılan `0.70`.
- `QUALITY_LOW_CONFIDENCE_PENALTY`: Varsayılan `0.20`.
- `QUALITY_INVALID_PENALTY`: Varsayılan `0.35`.
- `QUALITY_PARTIAL_THRESHOLD`: Varsayılan `0.55`.
- `AI_STORE_ANALYSES`: Varsayılan `True`.

`app/core/security.py` dosyası mevcut fakat boştur. Kodda authentication, authorization veya API key kontrolü uygulanmamıştır.

## 6. Docker ve Çalışma Ortamı

### 6.1 Dockerfile

Docker imajı `python:3.12-slim` tabanlıdır.

Kurulan sistem paketleri:

- `libmagic1`: MIME tespiti için.
- `libgl1`, `libglib2.0-0`: OpenCV için gerekli runtime kütüphaneleri.
- `curl`: Healthcheck için.
- `tesseract-ocr`: OCR motoru.
- `tesseract-ocr-eng`: İngilizce OCR dili.
- `tesseract-ocr-tur`: Türkçe OCR dili.

İmaj içinde:

- `/app` çalışma dizini kullanılır.
- `uv` binary’si resmi `ghcr.io/astral-sh/uv` imajından kopyalanır.
- `pyproject.toml` ve `uv.lock` kopyalanır.
- `uv sync --frozen --no-dev` ile bağımlılıklar kurulur.
- `app` dizini kopyalanır.
- `/storage/inputs` oluşturulur.
- `8000` portu expose edilir.
- Varsayılan komut `uvicorn app.main:app --host 0.0.0.0 --port 8000` çalıştırır.

### 6.2 docker-compose.yml

Compose içinde dört servis vardır:

- `api`: FastAPI servisi.
- `worker`: Celery worker.
- `postgres`: PostgreSQL 16.
- `redis`: Redis 7.

`api` servisi:

- `Dockerfile` ile build edilir.
- `.env` okur.
- Host `8000` portunu container `8000` portuna bağlar.
- `input_storage` volume’unu `/storage` altına bağlar.
- PostgreSQL ve Redis healthy olduktan sonra başlar.
- `/health/ready` endpointi ile healthcheck yapar.
- `restart: unless-stopped` kullanır.

`worker` servisi:

- Aynı imajı kullanır.
- Komut olarak Celery worker başlatır:
  - app: `app.workers.celery_app.celery_app`
  - queue: `stage2`
  - loglevel: `INFO`
- `/storage` volume’unu paylaşır.
- API ile aynı input dosyalarına erişebilir.

`postgres` servisi:

- Kullanıcı: `postgres`
- Şifre: `postgres`
- DB: `finance_ai`
- Host portu: `5433`
- Container portu: `5432`
- Healthcheck: `pg_isready -U postgres -d finance_ai`

`redis` servisi:

- Host portu: `6380`
- Container portu: `6379`
- Append-only mode açıktır.
- Healthcheck: `redis-cli ping`

### 6.3 docker-compose.dev.yml

Geliştirme compose dosyası hot reload destekler.

Farklar:

- API komutu `uvicorn app.main:app --reload` ile çalışır.
- `./app:/app/app` volume mount edilir.
- `./local_storage:/storage` kullanılır.
- API ve worker için LLM environment değerleri override edilir:
  - `LLM_ENABLED=true`
  - `LLM_BASE_URL=http://host.docker.internal:11434`
  - `LLM_MODEL=qwen3:8b`
- Container isimlerinde `_dev` suffix’i vardır.
- PostgreSQL ve Redis volume adları geliştirme için ayrıdır.

## 7. Veritabanı Katmanı

Veritabanı bağlantısı `app/db/session.py` içindedir.

- `create_engine()` `settings.DATABASE_URL` ile kurulur.
- `pool_pre_ping=True` kullanılır.
- `pool_size=5`, `max_overflow=10` ayarlanmıştır.
- `SessionLocal` autocommit ve autoflush kapalı şekilde oluşturulur.
- `get_db()` FastAPI dependency olarak session üretir ve finally bloğunda kapatır.

`app/db/base.py` içinde:

- Ana `Base`, `app.models.input_record` dosyasından import edilir.
- Diğer model modülleri side-effect olarak import edilir.
- `create_db_tables()` bütün metadata tablolarını oluşturur.

## 8. Veri Modelleri

### 8.1 InputRecord

Tablo: `inputs`

Amaç: Yüklenen orijinal dosyanın metadata kaydı.

Alanlar:

- `id`: Primary key. Format `inp_<uuidhex>`.
- `user_id`: Opsiyonel kullanıcı bilgisi.
- `original_filename`: Yüklenen dosyanın orijinal adı.
- `mime_type`: MIME tipi.
- `file_size`: Byte cinsinden dosya boyutu.
- `storage_key`: Lokal object storage key değeri.
- `storage_url`: Lokal dosya yolu.
- `status`: Varsayılan `uploaded`. Süreçte `classified`, `preprocessed`, `preprocessing_needs_review`, `preprocessing_failed` olabilir.
- `created_at`: UTC oluşturulma zamanı.
- `updated_at`: UTC güncelleme zamanı. Kodda otomatik update hook yoktur; default değer alır.

İlişki:

- `classification`: `InputClassification` ile bire bir ilişki.

### 8.2 InputClassification

Tablo: `input_classifications`

Amaç: Stage 1 sınıflandırma sonucunu saklamak.

Alanlar:

- `id`: Format `cls_<uuidhex>`.
- `input_id`: `inputs.id` foreign key.
- `kind`: Sınıflandırılan input tipi.
- `confidence`: 0-1 arası güven skoru.
- `needs_ocr`: OCR gerekip gerekmediği.
- `needs_preprocessing`: Ön işleme gerekip gerekmediği.
- `routing_key`: Celery routing key veya sonraki aşama işareti.
- `features_json`: Sınıflandırıcı tarafından çıkarılan özellikler.
- `warnings_json`: Uyarı listesi.
- `model_version`: Varsayılan `rules-v1`.
- `created_at`: UTC oluşturulma zamanı.

### 8.3 InputPreprocessingRecord

Tablo: `input_preprocessings`

Amaç: Stage 2 preprocessing sonucunu saklamak.

Alanlar:

- `id`: Format `prep_<uuidhex>`.
- `input_id`: `inputs.id` foreign key.
- `source_kind`: Input sınıfı.
- `status`: Varsayılan `completed`.
- `output_type`: Üretilen çıktı tipi.
- `output_storage_key`, `output_storage_url`: Çıktı kökü veya referansı.
- `preferred_output_storage_key`, `preferred_output_storage_url`: Extraction için önerilen çıktı.
- `preferred_output_variant`: Önerilen varyant adı.
- `preferred_extraction_method`: Varsayılan `ocr_multi_variant`.
- `extraction_risk`: Varsayılan `medium`.
- `page_count`: Sayfa sayısı.
- `preprocessing_version`: Ayarlardan gelen preprocessing sürümü.
- `operations_json`: Uygulanan işlem listesi.
- `quality_before_json`: İşlem öncesi kalite metrikleri.
- `quality_after_json`: İşlem sonrası kalite metrikleri.
- `outputs_json`: Sayfa/varyant çıktıları.
- `warnings_json`: Uyarılar.
- `average_quality_score_before`
- `average_quality_score_after`
- `ocr_readiness_score`
- `is_ready_for_extraction`: Extraction’a hazır mı.
- `created_at`

### 8.4 DataExtractionRecord

Tablo: `data_extractions`

Amaç: Stage 3 veri çıkarım sonucunu saklamak.

Alanlar:

- `id`: Format `ext_<uuidhex>`.
- `input_id`: `inputs.id` foreign key.
- `source_kind`: Kaynak input tipi.
- `extraction_type`: Örneğin `pdf_data_extraction` veya `image_ocr_data_extraction`.
- `extraction_method`: Örneğin `native_pdf_text` veya `tesseract_ocr_multi_variant`.
- `status`: Varsayılan `completed`.
- `extraction_version`: Varsayılan `pdf-native-v1`.
- `transaction_count`
- `low_confidence_count`
- `average_confidence`
- `result_json`: Extraction response gövdesi.
- `debug_json`: Debug bilgisi.
- `warnings_json`: Uyarılar.
- `created_at`

### 8.5 NormalizationRecord

Tablo: `normalizations`

Amaç: Stage 4 normalizasyon sonucunu saklamak.

Alanlar:

- `id`: Format `norm_<uuidhex>`.
- `input_id`: `inputs.id` foreign key.
- `extraction_id`: `data_extractions.id` foreign key.
- `status`: Varsayılan `completed`.
- `normalization_version`: Varsayılan `normalization-v1`.
- `transaction_count`
- `duplicate_removed_count`
- `low_confidence_count`
- `overall_confidence`
- `result_json`
- `scores_json`
- `warnings_json`
- `created_at`

### 8.6 AIAnalysisRecord

Tablo: `ai_analysis_records`

Amaç: AI analiz sonucunu saklamak.

Alanlar:

- `id`: Format `ai_<uuidhex>`.
- `input_id`: Index’li string alan.
- `status`: Varsayılan `completed`.
- `analysis_version`
- `llm_model`
- `llm_available`: String olarak `true`/`false`.
- `analysis_confidence`
- `request_json`
- `result_json`
- `warnings_json`
- `created_at`

## 9. API Endpointleri

### 9.1 Health Endpointleri

Dosya: `app/api/routes/health.py`

Endpointler:

- `GET /health`
- `GET /health/live`
- `GET /health/ready`

`/health`:

- Servis adını, environment bilgisini, genel status değerini ve component health sonuçlarını döndürür.
- PostgreSQL ve Redis durumunu `HealthService` ile kontrol eder.

`/health/live`:

- Sadece uygulamanın canlı olduğunu belirtir.
- Response örneği: `{"status": "ok", "service": "..."}`

`/health/ready`:

- PostgreSQL ve Redis hazır değilse HTTP 503 döndürür.
- Hazırsa status `ok` olur.

### 9.2 Input Upload ve Classification

Dosya: `app/api/routes/inputs.py`

Endpoint:

- `POST /v1/inputs`

Form alanları:

- `file`: Zorunlu dosya.
- `user_id`: Opsiyonel string.

İş akışı:

1. `UploadService.read_and_validate()` dosyayı okur.
2. Boş dosya ise 400 döner.
3. Maksimum dosya boyutunu aşarsa 413 döner.
4. MIME tipi `MimeDetector` ile tespit edilir.
5. Desteklenmeyen MIME tipi ise 415 döner.
6. `ObjectStorage.upload_bytes()` dosyayı lokal `/storage/inputs` altına yazar.
7. `InputRecord` oluşturulur ve status `uploaded` olur.
8. Dosya geçici path’e yazılır.
9. `ClassificationService.classify()` çağrılır.
10. Geçici dosya temizlenir.
11. `InputClassification` oluşturulur.
12. `InputRecord.status` `classified` yapılır.
13. `Stage2Dispatcher.dispatch()` ile Celery’ye `stage2.process_input` taskı gönderilir.
14. `InputUploadResponse` döner.

Response şeması:

- `input_id`
- `status`
- `classification`
- `next_stage`

### 9.3 Preprocessing Result

Dosya: `app/api/routes/preprocessing.py`

Endpoint:

- `GET /v1/preprocessings/{input_id}`

Davranış:

- Verilen input için en son `InputPreprocessingRecord` kaydını döndürür.
- Kayıt yoksa 404 `Preprocessing result not found` döner.
- Response içinde output bilgileri, tercih edilen extraction çıktısı, kalite metrikleri, readiness skoru, warnings ve preprocessing version bulunur.

### 9.4 Extraction Endpointleri

Dosya: `app/api/routes/extractions.py`

Endpointler:

- `POST /v1/extractions/pdf/{input_id}`
- `POST /v1/extractions/image/{input_id}`
- `GET /v1/extractions/{input_id}/latest`

`POST /v1/extractions/pdf/{input_id}`:

- Sadece `real_pdf` sınıfındaki inputlar için çalışır.
- Input yoksa 404 döner.
- Classification yoksa 409 döner.
- Classification kind `real_pdf` değilse 400 döner.
- En son preprocessing kaydı alınır.
- `preferred_extraction_method` `native_pdf_text` değilse 400 döner.
- `PdfExtractionService.extract()` ile text layer üzerinden işlem çıkarılır.
- `DataExtractionRecord` oluşturulur.
- Stage 1, Stage 2, Stage 3 metadata ve extraction result döndürülür.

`POST /v1/extractions/image/{input_id}`:

- Desteklenen kind değerleri:
  - `screenshot`
  - `camera_photo`
  - `scanned_pdf`
  - `hybrid_pdf`
- Classification yoksa 409 döner.
- Kind desteklenmiyorsa 400 döner.
- En son preprocessing kaydı alınır.
- `preferred_extraction_method` `ocr_multi_variant` değilse 400 döner.
- Preprocessing outputs üzerinden `ImageExtractionService.extract()` çağrılır.
- `DataExtractionRecord` oluşturulur.
- Stage bilgileri ve result döndürülür.

`GET /v1/extractions/{input_id}/latest`:

- En son extraction kaydını döndürür.
- Yoksa 404 `Extraction result not found`.

`include_debug` query parametresi:

- PDF ve image extraction endpointlerinde vardır.
- `true` olduğunda parser debug bilgileri response ve DB kaydına eklenir.

### 9.5 Normalization Endpointleri

Dosya: `app/api/routes/normalizations.py`

Endpointler:

- `POST /v1/normalizations/{input_id}`
- `GET /v1/normalizations/{input_id}/latest`

`POST /v1/normalizations/{input_id}`:

- Input için en son extraction kaydını alır.
- Extraction yoksa 404 döner.
- Extraction result içinde `transactions` yoksa nested `result` alanına bakar.
- `NormalizationService.normalize_extraction_result()` çağrılır.
- `NormalizationRecord` oluşturulur.
- `Stage4Response` döndürülür.

`GET /v1/normalizations/{input_id}/latest`:

- En son normalization kaydını döndürür.
- Yoksa 404 `Normalization result not found`.

### 9.6 AI Analysis Endpointleri

Dosya: `app/api/routes/analyze.py`

Prefix:

- `/v1/ai`

Endpointler:

- `POST /v1/ai/analyze`
- `POST /v1/ai/analyze-and-save`
- `GET /v1/ai/analyses/{input_id}/latest`
- `POST /v1/ai/chat`
- `GET /v1/ai/health`

`POST /v1/ai/analyze`:

- Normalize edilmiş transaction listesi ile analiz üretir.
- Hiç transaction yoksa 422 döner.
- `AIAnalysisService.analyze()` çağrılır.
- Sonucu DB’ye yazmaz.

`POST /v1/ai/analyze-and-save`:

- `analyze` ile aynı analizi yapar.
- Sonucu `AIAnalysisRecord` olarak DB’ye yazar.
- Response içindeki `analysis_id` alanına oluşturulan kayıt id’sini koyar.

`GET /v1/ai/analyses/{input_id}/latest`:

- Input için en son AI analiz kaydını döndürür.
- Yoksa 404 `AI analysis result not found`.

`POST /v1/ai/chat`:

- Kullanıcı sorusunu mevcut analiz bağlamında cevaplar.
- Request içinde doğrudan `analysis` verilebilir.
- Ya da `analysis_id` verilirse DB’den ilgili analiz okunur.
- İkisi de yoksa 422 döner.
- `LLMReportService.answer_question()` kullanılır.

`GET /v1/ai/health`:

- AI analiz katmanının servis durumlarını döndürür.
- Feature engineering, categorization, profiling, anomaly detection, forecast, installment ve chat enabled olarak listelenir.
- Embedding modelinin load edilebilir olup olmadığı kontrol edilir.
- Ollama provider availability kontrol edilir.

## 10. Pipeline Genel Akışı

Uçtan uca sistem şu sırayla çalışır:

1. Kullanıcı `/v1/inputs` endpointine dosya yükler.
2. Dosya MIME tipi tespit edilir ve lokal storage’a yazılır.
3. Dosya real PDF, scanned PDF, hybrid PDF, screenshot, camera photo, unknown veya unsupported olarak sınıflandırılır.
4. Classification sonucu DB’ye kaydedilir.
5. Celery’ye Stage 2 preprocessing taskı gönderilir.
6. Worker input dosyasını storage URL üzerinden okur.
7. Kaynak türüne göre preprocessing yapılır.
8. Preprocessing sonucu DB’ye yazılır.
9. Kullanıcı veya orchestrator uygun extraction endpointini çağırır.
10. Real PDF ise native text extraction, görsel/scanned PDF ise OCR extraction yapılır.
11. Extraction sonucu DB’ye yazılır.
12. Kullanıcı `/v1/normalizations/{input_id}` çağırır.
13. En son extraction sonucu normalize edilir, skorlanır ve DB’ye yazılır.
14. Kullanıcı normalize sonucu `/v1/ai/analyze` veya `/v1/ai/analyze-and-save` ile AI analizine gönderir.
15. AI servisleri kategori, profil, anomali, forecast, installment ve assistant sonuçları üretir.
16. İsteğe bağlı olarak analiz DB’ye kaydedilir.
17. Kullanıcı `/v1/ai/chat` ile analiz hakkında soru sorabilir.

## 11. Stage 1: Input Classification

### 11.1 Desteklenen MIME Tipleri

`MimeDetector.ALLOWED_MIME_TYPES`:

- `application/pdf`
- `image/jpeg`
- `image/png`
- `image/webp`
- `image/heic`
- `image/heif`

MIME tipi `python-magic` ile dosya bytes üzerinden tespit edilir.

### 11.2 InputKind Değerleri

`app/schemas/classification.py` içindeki enum:

- `real_pdf`
- `scanned_pdf`
- `hybrid_pdf`
- `screenshot`
- `camera_photo`
- `unknown`
- `unsupported`

### 11.3 Routing Key Eşleşmeleri

`RoutingService.get_routing_key()` mapping:

- `real_pdf`: `pdf.real.extract_text`
- `scanned_pdf`: `pdf.scanned.ocr`
- `hybrid_pdf`: `pdf.hybrid.extract_and_ocr`
- `screenshot`: `image.screenshot.ocr`
- `camera_photo`: `image.camera_photo.preprocessing`
- `unknown`: `manual_review`
- `unsupported`: `unsupported`

OCR gerektiren türler:

- `scanned_pdf`
- `hybrid_pdf`
- `screenshot`
- `camera_photo`

Preprocessing gerektiren türler:

- `scanned_pdf`
- `hybrid_pdf`
- `camera_photo`

Kodda screenshot için `needs_preprocessing=False` görünse de Stage 2 taskı yine dispatch edilir ve `PreprocessingService` screenshot için OCR varyantları üretir. Bu alan daha çok route metadata olarak kullanılır.

### 11.4 PDFClassifier

PDF sınıflandırıcı `PyMuPDF` ile PDF’i açar ve sayfa bazlı özellikler çıkarır.

Ölçülen metrikler:

- Sayfa sayısı.
- Toplam karakter sayısı.
- Toplam kelime sayısı.
- Text bulunan sayfa oranı.
- Büyük görsel içeren sayfa oranı.
- Global görsel alan oranı.
- İlk 5 sayfa için:
  - page number
  - char count
  - word count
  - image count
  - image area ratio

Eşikler:

- `MIN_TEXT_CHARS_FOR_TEXT_PAGE = 80`
- `STRONG_AVG_TEXT_THRESHOLD = 300`
- `LARGE_IMAGE_PAGE_RATIO_THRESHOLD = 0.60`
- `LARGE_IMAGE_AREA_THRESHOLD = 0.65`

Karar mantığı:

- Ortalama text karakteri 300 ve üzeri, text page ratio 0.60 ve üzeri ise `real_pdf`, confidence `0.94`.
- Ortalama text karakteri 80 altı ve large image page ratio 0.60 ve üzeri ise `scanned_pdf`, confidence `0.92`.
- Text page ratio 0.15 üstü ve large image page ratio 0.15 üstü ise `hybrid_pdf`, confidence `0.78`, warning `mixed_text_and_image_pdf`.
- Global image area ratio 0.50 üstü ve ortalama text karakteri 150 altı ise düşük güvenli `scanned_pdf`, confidence `0.74`, warning `likely_scanned_pdf_but_low_confidence`.
- Diğer durumlarda `unknown`, confidence `0.50`, warning `pdf_classification_uncertain`.

PDF açılamazsa veya sayfa sayısı sıfırsa `unknown` döner.

### 11.5 ImageClassifier

Görsel sınıflandırıcı iki grup özellik çıkarır:

EXIF özellikleri:

- EXIF var mı.
- Kamera ilişkili tag sayısı.
- Görsel genişlik/yükseklik.
- Aspect ratio.

OpenCV özellikleri:

- Blur score.
- Edge density.
- Belge konturu var mı.
- Maksimum contour area ratio.
- Quadrilateral count.
- Axis aligned line score.

Skorlama mantığı:

- PNG/WebP screenshot tarafına puan verir.
- JPEG/HEIC/HEIF camera photo tarafına puan verir.
- Kamera EXIF tag sayısı 2 veya daha fazlaysa camera photo kuvvetlenir.
- EXIF yoksa screenshot puanı artar.
- Çok yüksek blur score screenshot lehinedir.
- Düşük blur score camera photo lehinedir.
- Büyük belge konturu camera photo lehinedir.
- Axis aligned line score yüksekse screenshot lehinedir.
- Telefon screenshot aspect ratio aralıkları screenshot lehinedir.
- Yaygın mobil screenshot boyutları screenshot lehinedir.

Toplam skor sıfırsa `unknown` döner. Screenshot skoru camera skorundan yüksekse `screenshot`, aksi halde `camera_photo` döner. Confidence iki skorun oranından hesaplanır. Confidence 0.65 altındaysa `low_confidence_image_classification` uyarısı eklenir. Camera photo blur score 80 altındaysa `possible_blurry_camera_photo` uyarısı eklenir.

## 12. Lokal Storage

### 12.1 ObjectStorage

Dosya: `app/storage/object_storage.py`

Yüklenen orijinal dosyaları `settings.LOCAL_INPUT_STORAGE_DIR` altına yazar.

Davranış:

- Storage dizinini oluşturur.
- MIME tipinden dosya extension belirler.
- UUID tabanlı güvenli dosya adı üretir.
- Dosyayı diske yazar.
- `storage_key` formatı: `inputs/<uuid>.<ext>`
- `storage_url`: Lokal filesystem path.

### 12.2 ProcessedStorage

Dosya: `app/storage/processed_storage.py`

Preprocessing çıktısı olan görselleri `settings.LOCAL_PROCESSED_STORAGE_DIR` altına yazar.

`save_page_image()`:

- Path: `/storage/processed/<input_id>/page_0001_<variant>_<uuid>.png`
- Storage key: `processed/<input_id>/<filename>`

`save_single_image()`:

- Tekil görsel çıktılarını aynı input klasöründe saklar.

Variant isimleri regex ile sadeleştirilir; lowercase yapılır, güvenli karakterler dışındakiler `_` ile değiştirilir.

## 13. Celery Worker ve Stage 2 Dispatch

### 13.1 Celery App

Dosya: `app/workers/celery_app.py`

Celery app adı:

- `finance_ai_pipeline`

Broker ve backend:

- `settings.REDIS_URL`

Include:

- `app.workers.tasks`

Celery config:

- JSON serializer.
- JSON result serializer.
- Accepted content JSON.
- Timezone UTC.
- `task_track_started=True`.

### 13.2 Stage2Dispatcher

Dosya: `app/workers/stage2_dispatcher.py`

`dispatch()` metodu payload oluşturur:

- `input_id`
- `storage_key`
- `routing_key`
- `classification_kind`

Sonra Celery’ye şu taskı gönderir:

- Task name: `stage2.process_input`
- Queue: `stage2`
- Routing key: classification routing key.

### 13.3 stage2.process_input Taskı

Dosya: `app/workers/tasks.py`

Task akışı:

1. Payload’dan `input_id` ve `classification_kind` alınır.
2. DB session açılır.
3. `InputRecord` aranır.
4. Input yoksa error response döner.
5. `PreprocessingService.preprocess()` çağrılır.
6. `InputPreprocessingRecord` oluşturulur.
7. `is_ready_for_extraction=True` ise input status `preprocessed` yapılır.
8. Hazır değilse status `preprocessing_needs_review` yapılır.
9. DB commit edilir.
10. Özet response döner.

Exception durumunda:

- DB rollback yapılır.
- Input varsa status `preprocessing_failed` yapılır.
- Error response döner.
- Session kapatılır.

## 14. Stage 2: Preprocessing

Preprocessing servisinin amacı input türüne göre extraction’a en uygun çıktıyı üretmektir.

Ana dosya:

- `app/services/preprocessing/preprocessing_service.py`

Yardımcı dosyalar:

- `image_utils.py`
- `pdf_renderer.py`
- `variant_builder.py`
- `quality_analyzer.py`
- `ocr_readiness_analyzer.py`
- `preprocessors.py`

### 14.1 Kaynak Türe Göre Davranış

`PreprocessingService.preprocess()`:

- `real_pdf`: `_preprocess_real_pdf()`
- `scanned_pdf`, `hybrid_pdf`: `_preprocess_scanned_pdf()`
- `screenshot`, `camera_photo`: `_preprocess_single_image()`
- Diğerleri: `manual_review`, high risk, extraction’a hazır değil.

### 14.2 Real PDF Preprocessing

Real PDF için görüntüye çevirme veya OCR hazırlığı yapılmaz.

Sonuç:

- `output_type`: `native_pdf_reference`
- `preferred_output_variant`: `original_pdf`
- `preferred_extraction_method`: `native_pdf_text`
- `extraction_risk`: `low`
- `operations`:
  - `preserve_original_pdf`
  - `skip_image_rendering`
  - `skip_ocr`
- `quality_after`:
  - `native_pdf_available=True`
  - `page_count`
- `is_ready_for_extraction=True`

Sayfa sayısı `PDFRenderer.open_document()` ile bulunur.

### 14.3 Scanned/Hybrid PDF Preprocessing

Scanned ve hybrid PDF için:

1. PDF sayfaları `PDFRenderer.render_pages()` ile image’a çevrilir.
2. Her sayfa PIL’den OpenCV array’e çevrilir.
3. `VariantBuilder.build_scanned_pdf_variants()` çağrılır.
4. Her varyant PNG olarak processed storage’a yazılır.
5. OCR readiness score’a göre tercih edilen çıktı seçilir.

Çıktı:

- `output_type`: `multi_variant_full_page_images`
- `preferred_extraction_method`: `ocr_multi_variant`
- `extraction_risk`: `medium`

### 14.4 Screenshot Preprocessing

Screenshot için:

- `VariantBuilder.build_screenshot_variants()` kullanılır.
- Varyantlar:
  - `normalized_original`: Renk koruma/fallback.
  - `ocr_grayscale`: Grayscale + soft contrast, primary OCR adayı.
- `preferred_extraction_method`: `ocr_multi_variant`
- `extraction_risk`: `low`

### 14.5 Camera Photo Preprocessing

Camera photo için:

- `VariantBuilder.build_camera_photo_variants()` kullanılır.
- Varyantlar:
  - `normalized_original`
  - `enhanced_grayscale`
  - `thresholded`
- Denoise, contrast enhancement, deskew ve adaptive threshold uygulanır.
- `preferred_extraction_method`: `ocr_multi_variant`
- `extraction_risk`: `medium`

### 14.6 Varyant Seçimi

Her varyant `ImageVariant` dataclass ile temsil edilir:

- `variant`
- `purpose`
- `image`
- `operations`
- `quality`
- `is_preferred_candidate`
- `warnings`

Tercih mantığı:

- Önce `is_preferred_candidate=True` olan varyantlar değerlendirilir.
- Bu adaylar yoksa bütün varyantlar değerlendirilir.
- `quality["ocr_readiness_score"]` en yüksek olan seçilir.
- Output listesinde `is_preferred=True` olarak işaretlenir.
- Final preferred output, preferred işaretliler içinde en yüksek readiness score’a sahip olandır.

Readiness:

- Output yoksa extraction’a hazır değildir.
- Ortalama OCR score yoksa hazır kabul edilir.
- Ortalama OCR score `0.35` ve üzerindeyse hazır kabul edilir.

### 14.7 ImageUtils

Başlıca fonksiyonlar:

- `pil_to_cv()`: PIL görüntüyü EXIF transpose ederek OpenCV BGR array’e dönüştürür.
- `cv_to_pil()`: OpenCV array’i PIL’e çevirir.
- `read_image()`: Dosyadan PIL ile okuyup OpenCV array döndürür.
- `to_grayscale()`: Gri tonlamaya çevirir.
- `normalize_original()`: Gerekirse resize eder.
- `resize_if_too_large()`: En uzun kenarı belirli max değere indirir.
- `denoise()`: OpenCV fast non-local means denoise uygular.
- `enhance_contrast()`: CLAHE contrast enhancement uygular.
- `soft_enhance_grayscale()`: Daha hafif CLAHE uygular.
- `adaptive_threshold()`: Gaussian blur + adaptive threshold.
- `otsu_threshold()`: Otsu threshold.
- `remove_small_noise()`: Morphological open ile küçük noise temizler.
- `deskew()`: Otsu binary üzerinden min area rect ile eğim düzeltir.
- `encode_png()`: OpenCV image’ı PNG bytes’a çevirir.

### 14.8 QualityAnalyzer

Görsel kalite metrikleri:

- `width`
- `height`
- `blur_score`
- `brightness`
- `contrast`
- `edge_density`
- `dark_pixel_ratio`
- `bright_pixel_ratio`
- `quality_score`

Quality score bileşenleri:

- Blur.
- Brightness.
- Contrast.
- Edge density.
- Dark/bright pixel oranları.

### 14.9 OcrReadinessAnalyzer

OCR’a hazır olma skoru üretir.

Metrikler:

- `blur_score`
- `brightness`
- `contrast`
- `foreground_density`
- `component_count`
- `avg_component_area`
- `border_artifact_score`
- `ocr_readiness_score`

Skor; netlik, parlaklık, kontrast, foreground density, connected component sayısı, component boyutu ve kenar artifact yoğunluğuna göre hesaplanır.

### 14.10 preprocessors.py

Bu dosyada eski/alternatif preprocessor sınıfları bulunur:

- `ScreenshotPreprocessor`
- `CameraPhotoPreprocessor`
- `ScannedPdfPagePreprocessor`
- `RealPdfPagePreprocessor`

Mevcut `PreprocessingService`, doğrudan bu sınıfları kullanmak yerine `VariantBuilder` merkezli yeni varyant üretim akışını kullanır. Yine de dosya, proje içinde tekrar kullanılabilecek spesifik preprocessing stratejilerini barındırır.

## 15. Stage 3: Data Extraction

Extraction iki ana kola ayrılır:

- Native PDF text extraction.
- OCR multi-variant extraction.

Ortak transaction output şeması:

- `date`
- `description`
- `price`
- `currency`
- `original_price`
- `original_currency`
- `installment`
- `direction`
- `confidence`
- `page`

Direction değerleri:

- `debit`
- `credit`
- `unknown`

### 15.1 Native PDF Extraction

Ana servis:

- `app/services/extraction/pdf_extraction_service.py`

Akış:

1. `PdfTextExtractor.extract_lines()` ile PDF text satırları çıkarılır.
2. `PdfTransactionParser.build_candidate_lines()` ile transaction adayı satırlar hazırlanır.
3. Doküman para birimi `infer_document_currency()` ile bulunur.
4. Her candidate line `parse_line_as_transaction()` ile parse edilir.
5. Başarılı transactionlar listeye eklenir.
6. Debug açıksa parsed ve rejected satırlar toplanır.
7. Summary hesaplanır.
8. `PdfExtractionResult` döndürülür.

Warnings:

- Text satırı yoksa `no_text_lines_extracted_from_pdf`.
- Transaction bulunamazsa `no_transactions_detected`.

Summary:

- `transaction_count`
- `low_confidence_count`
- `average_confidence`
- `total_debit`
- `total_credit`
- `document_currency`

### 15.2 PdfTextExtractor

Önce `pdfplumber` denenir.

`_extract_with_pdfplumber()`:

- Her sayfadan `extract_text(x_tolerance=1, y_tolerance=3, layout=False)` ile text alır.
- Satırları böler.
- `remove_pdf_artifacts()` uygular.
- Boş satırları atar.
- Satırlara source `pdfplumber` koyar.

Eğer pdfplumber hiç satır çıkaramazsa `PyMuPDF` fallback çalışır.

`_extract_with_pymupdf()`:

- `page.get_text("text")` ile text alır.
- Aynı temizleme ve satır üretimi yapılır.
- Source `pymupdf` olur.

### 15.3 PdfTransactionParser

Transaction parse mantığı bu sınıfta merkezileşmiştir ve OCR parser tarafından da inherit edilir.

Önemli yetenekler:

- Tarih bulma.
- Para değeri bulma.
- Taksit bilgisi algılama.
- Gürültü satırlarını eleme.
- Açıklama temizleme.
- İşlem tutarı/orijinal tutar seçme.
- Direction çıkarma.
- Confidence hesaplama.

Candidate line oluşturma:

- Orijinal satırlar korunur.
- Satırda tarih var ama para yoksa aynı sayfadaki sonraki 1-2 satırla merge denenir.
- Bu, bazı PDF layoutlarında transaction bilgisinin birkaç satıra bölünmesine tolerans sağlar.

Tarih formatları:

- Türkçe ay isimli tarih: `1 Ocak 2025`
- Gün/ay/yıl: `01.01.2025`, `01/01/2025`, `01-01-2025`
- ISO benzeri: `2025-01-01`

Para formatları:

- Currency önce veya sonra olabilir.
- `TL`, `TRY`, `₺`, `$`, `USD`, `€`, `EUR` ve `pycountry` üzerinden ISO para birimleri desteklenir.
- Türkçe ve uluslararası binlik/ondalık ayrımlar normalize edilir.

Taksit formatları:

- `1. Taksit`
- `(1/6)`
- `100,00 x 6 = 600,00`

Noise keywords:

- Kart limiti.
- Nakit avans limiti.
- Dönem borcu.
- Asgari ödeme.
- Hesap özeti.
- Son ödeme.
- Ekstre no.
- Kart numarası.
- Banka/kurumsal metinler.

Direction:

- Metinde kredi/iade/ödeme gibi kelimeler varsa `credit`.
- Aksi halde varsayılan `debit`.

Confidence:

- Tarih: 0.35
- Tutar: 0.30
- Açıklama: 0.20
- Currency: 0.10
- Taksit current varsa: 0.03
- Maksimum 0.98.

### 15.4 OCR Extraction

Ana servis:

- `app/services/extraction/image_extraction_service.py`

Akış:

1. Preprocessing outputs sayfa numarasına göre gruplanır.
2. Her sayfa için `OcrVariantSelector.select_best_for_page()` çağrılır.
3. En iyi varyantın OCR satırları alınır.
4. Tüm sayfalardaki OCR satırlarından doküman para birimi çıkarılır.
5. Satırlar `OcrTransactionParser.parse_ocr_line_as_transaction()` ile parse edilir.
6. Debug açıksa variant selection, rejected lines ve parsed source lines toplanır.
7. Summary ve warnings üretilir.

Warnings:

- Preprocessing output yoksa `no_preprocessing_outputs_found`.
- OCR satırı yoksa `no_ocr_lines_extracted`.
- Transaction yoksa `no_transactions_detected`.

### 15.5 OcrTextExtractor

Tesseract kullanır.

`extract_words()`:

- PIL ile image açar.
- Config: `--oem 3 --psm <psm> -c preserve_interword_spaces=1`
- Varsayılan lang: `tur+eng`.
- `pytesseract.image_to_data()` ile kelime ve bounding box bilgisi alır.
- Confidence negatif olanları atar.
- Confidence 0-1 aralığına normalize edilir.

`words_to_lines()`:

- Kelimeleri y ve x koordinatına göre sıralar.
- Median kelime yüksekliğine göre y tolerance hesaplar.
- Benzer y merkezindeki kelimeleri satır grubuna toplar.
- Satır text, bounding box, ortalama OCR confidence ve word count üretir.

`extract_lines()`:

- `extract_words()` ve `words_to_lines()` işlemlerini birleştirir.

### 15.6 OcrVariantSelector

Her sayfa için en iyi OCR varyantını seçer.

Denediği PSM değerleri:

- `6`
- `11`
- `4`

Variant priority örnekleri:

- `enhanced_grayscale`: 100
- `ocr_grayscale`: 100
- `rendered_original`: 90
- `normalized_original`: 90
- `thresholded`: 55
- `binary`: 50

Final score:

- OCR line score.
- Variant priority / 100.
- Preprocessing preferred output ise ek `0.15`.

OCR line score bileşenleri:

- Transaction candidate count * 8.
- Date count * 2.
- Money count * 1.5.
- Ortalama OCR confidence * 15.
- Text length bonus.

En yüksek final score’a sahip sonuç seçilir.

### 15.7 OcrTransactionParser

`PdfTransactionParser` sınıfından türetilmiştir.

Davranış:

- OCR line, PDF parser’ın beklediği line benzeri adapter objesine dönüştürülür.
- Base parse işlemi aynen kullanılır.
- Son confidence OCR confidence ve word count ile tekrar hesaplanır.

OCR confidence formülü:

- Base confidence * 0.78
- OCR confidence * 0.18
- Word count 3 veya üstü ise +0.04
- Maksimum 0.98.

## 16. Stage 4: Normalization

Ana servis:

- `app/services/normalization/normalization_service.py`

Amaç:

- Extraction sonucundaki transactionları standart formata çevirmek.
- Merchant bilgisini normalize etmek.
- Amount/currency/date alanlarını doğrulamak.
- Duplicate kayıtları temizlemek.
- Satır bazlı ve genel confidence skorları üretmek.

### 16.1 NormalizationService Akışı

1. Extraction result içinden `transactions` alınır.
2. `TransactionNormalizer.normalize_many()` çağrılır.
3. Her transaction için `ScoringService.score_transaction()` çalışır.
4. Transaction confidence, row score ile değiştirilir.
5. `ScoringService.summarize()` genel skor özetini çıkarır.
6. `TransactionNormalizer.build_summary()` finansal summary üretir.
7. Extraction debug bilgisinden normalized debug oluşturulur.
8. Duplicate silinmişse warnings’e `duplicates_removed:<count>` eklenir.
9. `Stage4Response` döndürülür.

### 16.2 TransactionNormalizer

Normalize edilen alanlar:

- `date`: ISO date formatına çevrilir.
- `amount`: Decimal ile iki basamağa yuvarlanır.
- `currency`: Uppercase yapılır, `TL` ise `TRY`.
- `original_amount`
- `original_currency`
- `description`: Whitespace ve kenar karakterleri temizlenir.
- `merchant`: `MerchantNormalizer` ile normalize edilir.
- `direction`: Geçerli değilse `unknown`.
- `installment`: Integer ve amount alanları normalize edilir.

Validation uyarıları:

- `missing_or_invalid_date`
- `missing_amount`
- `non_positive_amount`
- `missing_currency`
- `invalid_currency`
- `missing_or_short_description`
- `invalid_installment_current_gt_total`

Validation status:

- Hard failure varsa `invalid`.
- Uyarı varsa `warning`.
- Uyarı yoksa `valid`.

Hard failure uyarıları:

- `missing_or_invalid_date`
- `missing_amount`
- `non_positive_amount`

Transaction ID:

- Date, merchant, amount, currency ve direction birleşimi SHA-256 ile hashlenir.
- İlk 16 hex karakter alınır.
- Format: `txn_<digest>`.

Duplicate key:

- Date.
- Normalized merchant.
- Amount.
- Currency.
- Direction.

Aynı duplicate key’e sahip kayıtlar arasında confidence yüksek olan tutulur.

Summary:

- `transaction_count`
- `duplicate_removed_count`
- `total_debit`
- `total_credit`
- `net_amount`
- `currencies`
- `primary_currency`
- `low_confidence_count`
- `invalid_count`
- `warning_count`
- `average_confidence`

### 16.3 MerchantNormalizer

Merchant normalize etme adımları:

- Unicode NFKC normalize edilir.
- Non-breaking space temizlenir.
- Türkçe `İ` ve `ı` karakterleri sadeleştirilir.
- Whitespace sadeleştirilir.
- Kenar ayırıcı karakterler temizlenir.
- Uppercase canonical form üretilir.
- Bazı bilinen aliaslar canonical merchant’a çevrilir.
- Uzun sayısal referanslar temizlenir.

Bilinen alias örnekleri:

- `OBILET`, `OBİLET` -> `OBILET.COM`
- `IYZICO AMAZON`, `IYZICO *AMAZON` -> `IYZICO *AMAZON.COM`
- `BKMKITAP` -> `IYZICO/BKMKITAP.COM`
- `PEGASUS`
- `MEDIA MARKT`
- `TRENDYOL`
- `STEAM`

Merchant confidence:

- Boşsa 0.20.
- Base 0.60.
- Normalized uzunluğu 3 ve üzeriyse +0.20.
- Alphabetic karakter içeriyorsa +0.10.
- Known alias canonical değerlerinden biriyse +0.10.
- Maksimum 0.98.

### 16.4 ScoringService

Satır skoru bileşenleri:

- Extraction confidence: %40.
- Field confidence: %25.
- Completeness score: %20.
- Validation score: %15.

Field confidence:

- Date: 0.25
- Amount: 0.25
- Currency: 0.15
- Description: 0.15
- Merchant confidence: 0.20 ağırlıklı.

Completeness score:

- Date, description, amount, currency, direction alanlarının doluluk oranı.

Validation score:

- `valid`: 1.0
- `warning`: 0.65
- `invalid`: 0.25

Flags:

- Skor threshold altındaysa `low_confidence`.
- Validation invalid ise `validation_failed`.
- Validation warning ise `validation_warning`.

Varsayılan low confidence threshold:

- `0.70`

Score summary:

- `overall_confidence`
- `min_confidence`
- `max_confidence`
- `low_confidence_threshold`
- `low_confidence_count`
- `invalid_count`
- `warning_count`
- `validation_passed`

## 17. Stage 5: AI Analysis

Ana servis:

- `app/services/ai/analysis_service.py`

AI analysis, normalize edilmiş transactionlar üzerinden finansal içgörü üretir.

Alt servisler:

- `FeatureEngineeringService`
- `CategorizationService`
- `SpendingProfileService`
- `AnomalyDetectionService`
- `ForecastInstallmentService`
- `LLMReportService`
- `OllamaProvider`
- `EmbeddingCategoryClassifier`

### 17.1 AnalyzeRequest

Request alanları:

- `input_id`
- `status`
- `result`: Normalize pipeline sonucu.
- `scores`: Normalization skorları.
- `warnings`: Önceki aşamalardan gelen uyarılar.
- `historical_transactions`: Geçmiş transaction listesi.
- `question`: Opsiyonel kullanıcı sorusu.
- `purchase_scenario`: Opsiyonel taksit senaryosu.
- `use_llm`: Varsayılan `True`.

`result.transactions` boşsa API 422 döndürür.

### 17.2 AIAnalysisService Akışı

1. Normalize transactionlar ve historical transactionlar dataframe’e çevrilir.
2. `use_llm`, request değeri ve global `settings.LLM_ENABLED` ile belirlenir.
3. Ollama erişilebilirliği kontrol edilir.
4. Kategorilendirme yapılır.
5. Harcama profili çıkarılır.
6. Anomali tespiti yapılır.
7. Ana currency belirlenir.
8. Harcama forecast üretilir.
9. Purchase scenario varsa taksit önerisi üretilir.
10. Question varsa assistant cevabı üretilir.
11. Executive summary üretilir.
12. Quality hesaplanır.
13. Warnings birleştirilir.
14. Status `completed` veya `partial` seçilir.
15. `AnalyzeResponse` döndürülür.

Status `partial` olur:

- Invalid transaction varsa.
- Analysis confidence `QUALITY_PARTIAL_THRESHOLD` değerinin altındaysa.

### 17.3 FeatureEngineeringService

Transactionları pandas dataframe’e çevirir.

Üretilen kolonlar:

- `transaction_id`
- `dataset_role`
- `date`
- `description`
- `merchant`
- `amount`
- `currency`
- `original_amount`
- `original_currency`
- `direction`
- `confidence`
- `validation_status`
- `installment_current`
- `installment_total`
- `has_installment`
- `date_dt`
- `month_dt`
- `month`
- `weekday`
- `is_weekend`
- `spend_amount`
- `credit_amount`
- `signed_amount`
- `log_amount`
- `is_foreign_currency`
- `is_low_confidence`
- `is_invalid`

Current ve historical data birleştirilirken duplicate `transaction_id` değerlerinde son kayıt tutulur.

### 17.4 CategorizationService

Kategori üretim sırası:

1. Direction `credit` ise doğrudan `income_or_refund`.
2. Merchant/text rule matching.
3. Embedding similarity.
4. LLM fallback.
5. `other` fallback.

Kategori sonuçları yerelleştirilir. Örneğin:

- `groceries` -> `Market`
- `fuel` -> `Akaryakıt`
- `telecom` -> `İletişim`

Summary sadece debit transactionlar üzerinden oluşturulur.

Response metrikleri:

- `transactions`
- `summary`
- `uncategorized_count`
- `rule_assisted_count`
- `embedding_assisted_count`
- `llm_assisted_count`

### 17.5 Kategori Taksonomisi

Dosya:

- `app/services/ai/resources/category_taxonomy.yaml`

Taksonomi YAML olarak tutulur. Her kategori:

- `id`
- `label`
- `subcategories`
- `examples`
- `patterns`

Mevcut ana kategoriler:

- `travel`: Seyahat
- `food`: Yeme İçme
- `groceries`: Market
- `fuel`: Akaryakıt
- `transport`: Ulaşım
- `health`: Sağlık
- `clothing`: Giyim
- `entertainment`: Eğlence
- `telecom`: İletişim
- `subscription`: Abonelik
- `shopping`: Alışveriş
- `accommodation`: Konaklama
- `education`: Eğitim
- `insurance`: Sigorta
- `utilities_tax`: Fatura ve Vergi
- `finance`: Bankacılık
- `payment`: Kart Ödemesi
- `other`: Diğer

Taksonomi hem regex rule matching hem embedding reference document üretimi için kullanılır.

### 17.6 EmbeddingCategoryClassifier

Embedding modeli:

- Varsayılan `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Davranış:

- `EMBEDDING_ENABLED=False` ise çalışmaz.
- Merchant text boşsa çalışmaz.
- Model lazy load edilir.
- Taksonomiden embedding reference document listesi üretilir.
- Query ve reference embeddingleri cosine similarity ile karşılaştırılır.
- En iyi similarity `EMBEDDING_SIMILARITY_THRESHOLD` altındaysa sonuç dönmez.
- Üstündeyse category/subcategory prediction döner.

`predict_topk()` LLM promptuna olası kategori ipuçları sağlamak için kullanılır.

### 17.7 SpendingProfileService

Debit transactionlar üzerinden profil çıkarır.

Metrikler:

- Primary category.
- Primary category share.
- Installment transaction ratio.
- Foreign currency transaction ratio.
- Average transaction amount.
- Largest transaction amount.

Label üretimi:

- Primary category share >= 0.40 ise `<category>_heavy_spender`.
- Installment ratio >= 0.35 ise `installment_heavy_spender`.
- Foreign currency ratio >= 0.20 ise `international_spender`.
- Largest amount average amount’un 3 katından fazlaysa ve transaction count >= 3 ise `large_purchase_sensitive`.
- Hiçbiri yoksa `balanced_spender`.

Debit transaction yoksa:

- Label: `insufficient_spending_data`
- Observation: analiz için harcama işlemi bulunamadı.

### 17.8 AnomalyDetectionService

Anomali tespiti debit transactionlar üzerinde yapılır.

Yeterli reference data varsa:

- Minimum satır sayısı `ANOMALY_MIN_ROWS_FOR_PYOD`.
- Varsayılan 8.
- `pyod.models.ecod.ECOD` kullanılır.
- Features:
  - `log_amount`
  - `has_installment`
  - `is_foreign_currency`
  - `is_low_confidence`
  - `is_weekend`
- Features `MinMaxScaler` ile normalize edilir.
- ECOD decision score normalize edilir.
- Business flags bonus olarak eklenir.
- Score `ANOMALY_PYOD_SCORE_CUTOFF` altındaysa item üretilmez.

Yeterli reference data yoksa:

- Robust statistical fallback kullanılır.
- Median ve MAD üzerinden unusually high amount aranır.
- Business flags yine bonus verir.
- Score `ANOMALY_ROBUST_SCORE_CUTOFF` altındaysa item üretilmez.

Business flags:

- `foreign_currency_transaction`: +0.12
- `installment_transaction`: +0.05
- `low_source_confidence`: +0.18
- `source_validation_warning`: +0.18

Severity:

- Score >= 0.70: `high`
- Score >= 0.45: `medium`
- Aksi: `low`

LLM açıksa ve anomaly varsa kısa açıklama üretilebilir. LLM açıklama üretmezse deterministic mesajlar kullanılır.

### 17.9 ForecastInstallmentService

İki işi vardır:

- Aylık harcama forecast.
- Purchase scenario için taksit önerisi.

Forecast:

- Sadece debit, geçerli, aynı currency işlemler değerlendirilir.
- Aylık toplam harcama serisi oluşturulur.
- Eksik aylar 0 ile doldurulur.

Veri yoksa:

- Status `insufficient_data`.
- Method `no_data`.

Ay sayısı Transformer için yetersizse:

- Method `weighted_moving_average_fallback_v2`.
- Recent window ağırlıklı hareketli ortalama kullanılır.
- Confidence `0.35 + month_count * 0.06`, maksimum `0.65`.

Ay sayısı yeterliyse:

- Method `transformer_moving_average_ensemble_v1`.
- Küçük bir `MonthlySpendTransformer` eğitilir.
- Transformer prediction ve weighted moving average harmanlanır.
- Blend: %60 transformer, %40 weighted average.
- Prediction geçmiş min/max aralığına clamp edilir.
- Confidence `0.65 + month_count * 0.025`, maksimum `0.90`.

Transformer:

- `d_model=32`
- `nhead=4`
- `num_layers=2`
- Feedforward `64`
- Activation `gelu`
- Output son token üzerinden alınır.
- Eğitim epoch sayısı `FORECAST_TRAIN_EPOCHS`, varsayılan 120.
- Seed ayarı yapılır.

Taksit önerisi:

- `purchase_scenario` yoksa status `not_requested`.
- Forecast tamamlanmamışsa veya currency uyuşmuyorsa status `insufficient_data`.
- Her ay sayısı için monthly amount ve burden ratio hesaplanır.
- Risk:
  - Burden ratio <= 0.15: `low`
  - Burden ratio <= 0.30: `medium`
  - Aksi: `high`
- İlk low risk option önerilir.
- Low yoksa en uzun vade önerilir.
- LLM varsa açıklama LLM ile üretilebilir.
- Warning olarak `recommendation_is_spending_burden_estimate_not_credit_advice` eklenir.

### 17.10 LLMReportService

Üç ana işlevi vardır:

- Executive summary üretmek.
- Kullanıcı sorusunu cevaplamak.
- Deterministic fallback sağlamak.

Executive summary:

- Önce deterministic bullet/fact listesi üretir.
- LLM yoksa bu liste döner.
- LLM varsa doğrulanmış bulguları kısa Türkçe özet paragrafına dönüştürür.
- LLM’e yeni tutar veya bulgu icat etmemesi söylenir.

Question answering:

- Soru yoksa boş assistant cevabı döner.
- Intent tespiti basit keyword ile yapılır:
  - `category_question`
  - `anomaly_question`
  - `installment_question`
  - `forecast_question`
  - `general_statement_question`
- Önce deterministic cevap hazırlanır.
- LLM yoksa deterministic cevap döner.
- LLM varsa sadece analiz bağlamı kullanılarak kısa Türkçe cevap üretilir.

Chat context:

- En yoğun kategori.
- Profil label’ları.
- İlk 3 kategori.
- Anomali sayısı ve ilk 3 anomali.
- Forecast değeri.
- Taksit önerisi.

### 17.11 OllamaProvider

Dosya:

- `app/services/ai/providers/ollama_provider.py`

Availability:

- `LLM_ENABLED=False` ise unavailable.
- Aksi halde `GET <base_url>/api/tags` denenir.

Text generation:

- Endpoint: `POST <base_url>/api/chat`
- `stream=False`
- `think=False`
- `keep_alive=settings.LLM_KEEP_ALIVE`
- Messages: system + user
- Options:
  - temperature
  - num_ctx
  - seed
  - top_p
  - top_k
  - repeat_penalty
  - num_predict

Structured generation:

- Response Pydantic modelinin JSON schema’sı Ollama `format` alanına verilir.
- Prompt, yalnızca geçerli JSON döndürmesini ister.
- Validation başarısızsa `LLM_MAX_RETRIES + 1` kez dener.
- Sonuç model validate edilemezse `None` döner.

## 18. Pydantic Şemaları

### 18.1 Classification

`ClassificationResult`:

- `kind`
- `confidence`
- `needs_ocr`
- `needs_preprocessing`
- `routing_key`
- `features`
- `warnings`
- `model_version`

### 18.2 Input Response

`InputUploadResponse`:

- `input_id`
- `status`
- `classification`
- `next_stage`

### 18.3 Preprocessing

`PreprocessedPageOutput`:

- `page_number`
- `variant`
- `purpose`
- `is_preferred`
- `storage_key`
- `storage_url`
- `width`
- `height`
- `operations`
- `quality_before`
- `quality_after`
- `warnings`

`PreprocessingResult`:

- Input, source, status ve output metadata.
- Preferred output bilgileri.
- Preferred extraction method.
- Extraction risk.
- Page count.
- Outputs.
- Operations.
- Quality before/after.
- Warnings.
- Average quality scores.
- OCR readiness score.
- Ready flag.
- Preprocessing version.

### 18.4 Extraction

`ExtractedTransaction`:

- `date`
- `description`
- `price`
- `currency`
- `original_price`
- `original_currency`
- `installment`
- `direction`
- `confidence`
- `page`

`PdfExtractionResult`:

- `transactions`
- `summary`
- `debug`
- `warnings`

`PdfExtractionApiResponse`:

- `input_id`
- `status`
- `stage1`
- `stage2`
- `stage3`
- `result`

### 18.5 Normalization

`NormalizedTransaction`:

- `transaction_id`
- `date`
- `description`
- `merchant`
- `amount`
- `currency`
- `original_amount`
- `original_currency`
- `direction`
- `installment`
- `source`
- `confidence`
- `validation_status`
- `warnings`

`Stage4Response`:

- `input_id`
- `status`
- `result`
- `scores`
- `warnings`

### 18.6 Analyze

`AnalyzeRequest`:

- Normalized pipeline result.
- Scores.
- Warnings.
- Historical transactions.
- Question.
- Purchase scenario.
- use_llm flag.

`AnalyzeResponse`:

- `input_id`
- `analysis_id`
- `status`
- `result`
- `quality`
- `engine`
- `warnings`

`AiAnalysisResult`:

- Categorization.
- Spending profile.
- Anomalies.
- Forecast.
- Installment recommendation.
- Assistant answer.
- Executive summary.

## 19. Health Service

`HealthService` PostgreSQL ve Redis’i kontrol eder.

Database check:

- SQLAlchemy engine ile connection açar.
- `SELECT 1` çalıştırır.
- Başarılıysa component `postgresql`, status `ok`.
- Hata olursa status `error` ve hata mesajı.

Redis check:

- `Redis.from_url()` ile bağlanır.
- `ping()` çalıştırır.
- Başarılıysa component `redis`, status `ok`.
- Hata olursa status `error` ve hata mesajı.

Readiness:

- Database ve Redis componentleri toplanır.
- Hepsi `ok` ise genel status `ok`.
- Herhangi biri hatalıysa genel status `error`.

## 20. Test Yapısı

Test dizini:

- `tests/conftest.py`
- `tests/test_health.py`
- `tests/test_analyze_contract.py`
- `tests/fixtures/`

Fixture dosyaları:

- `real_pdf.pdf`
- `scanned_pdf.pdf`
- `screenshot.jpg`
- `camera_photo.jpeg`

`conftest.py`:

- Her test için geçici storage root oluşturur.
- `LOCAL_STORAGE_ROOT`, `LOCAL_INPUT_STORAGE_DIR`, `CLASSIFICATION_MODEL_VERSION` environment değerlerini monkeypatch eder.

`test_health.py`:

- FastAPI `TestClient` ile `/health/live` endpointini çağırır.
- Status code 200 ve response status `ok` bekler.

`test_analyze_contract.py`:

- External LLM ve embedding modelleri kapatılır.
- Örnek transaction seti oluşturulur:
  - MIGROS alışveriş.
  - SHELL akaryakıt.
  - TURKCELL fatura.
  - Maaş ödemesi.
- `AIAnalysisService().analyze()` çağrılır.
- Response’un `AnalyzeResponse` contractına uyduğu test edilir.
- Core alanların dolu olduğu test edilir.
- Rule tabanlı kategori eşleşmeleri doğrulanır:
  - MIGROS -> Market
  - SHELL -> Akaryakıt
  - TURKCELL -> İletişim

Mevcut test kapsamı ağırlıklı olarak health endpointi ve AI analysis contract/kategori kuralları üzerinedir. Upload, preprocessing, OCR extraction, PDF extraction, normalization ve Celery worker akışları için test fixture dosyaları bulunmasına rağmen mevcut test dosyalarında uçtan uca aktif testler görülmemektedir.

## 21. Çalıştırma

README içinde lokal geliştirme için şu komutlar verilmiştir:

```bash
uv sync --all-groups
uv run uvicorn app.main:app --reload
```

Docker Compose ile çalıştırma için beklenen akış:

```bash
docker compose up --build
```

Geliştirme compose dosyasıyla:

```bash
docker compose -f docker-compose.dev.yml up --build
```

API varsayılan olarak:

- `http://localhost:8000`

PostgreSQL host portu:

- `5433`

Redis host portu:

- `6380`

## 22. Örnek Kullanım Akışı

1. Dosya yükle:

```http
POST /v1/inputs
Content-Type: multipart/form-data
```

Form:

- `file=<pdf veya image>`
- `user_id=<opsiyonel>`

2. Preprocessing sonucunu kontrol et:

```http
GET /v1/preprocessings/{input_id}
```

3. Classification real PDF ise native extraction çağır:

```http
POST /v1/extractions/pdf/{input_id}?include_debug=false
```

4. Classification screenshot, camera photo, scanned PDF veya hybrid PDF ise OCR extraction çağır:

```http
POST /v1/extractions/image/{input_id}?include_debug=false
```

5. Normalization çalıştır:

```http
POST /v1/normalizations/{input_id}
```

6. En son normalization sonucunu al:

```http
GET /v1/normalizations/{input_id}/latest
```

7. AI analysis çalıştır:

```http
POST /v1/ai/analyze
```

veya DB’ye kaydetmek için:

```http
POST /v1/ai/analyze-and-save
```

8. Analiz hakkında soru sor:

```http
POST /v1/ai/chat
```

## 23. Hata Durumları ve Uyarılar

Yaygın HTTP hata durumları:

- Upload boş dosya: 400 `Empty file`
- Upload boyut limiti aşımı: 413 `File too large`
- Desteklenmeyen MIME tipi: 415 `Unsupported file type`
- Input bulunamadı: 404 `Input not found`
- Classification yok: 409 `Input has not been classified yet`
- Preprocessing yok: 409 veya 404, endpoint bağlamına göre.
- Preprocessing extraction’a hazır değil: 409.
- Yanlış extraction endpointi: 400.
- AI analysis transaction yok: 422.
- Chat için analysis/analysis_id yok: 422.
- AI analysis kaydı yok: 404.

Pipeline warning örnekleri:

- `unsupported_mime_type`
- `mixed_text_and_image_pdf`
- `likely_scanned_pdf_but_low_confidence`
- `pdf_classification_uncertain`
- `low_confidence_image_classification`
- `possible_blurry_camera_photo`
- `unsupported_source_kind_for_preprocessing`
- `no_text_lines_extracted_from_pdf`
- `no_transactions_detected`
- `no_preprocessing_outputs_found`
- `no_ocr_lines_extracted`
- `duplicates_removed:<count>`
- `analysis_contains_low_confidence_transactions`
- `analysis_contains_invalid_transactions`
- `llm_unavailable_deterministic_fallback_used`
- `recommendation_is_spending_burden_estimate_not_credit_advice`

## 24. Güçlü Yanlar

- Çok aşamalı pipeline net ayrılmıştır.
- API, worker, storage, DB ve AI servisleri modülerdir.
- PDF ve OCR extraction aynı transaction parser mantığını paylaşır.
- OCR için çoklu varyant ve PSM seçimi yapılır.
- Her aşama DB’ye ayrı kayıt bıraktığı için takip edilebilirlik yüksektir.
- LLM opsiyoneldir; sistem deterministic fallback ile çalışmaya devam eder.
- Kategori kuralları YAML taksonomi üzerinden genişletilebilir.
- Normalization aşamasında duplicate temizleme ve kalite skorlaması vardır.
- AI analysis yalnızca LLM’e bağlı değildir; profil, anomali ve forecast deterministic/statistical olarak üretilebilir.

## 25. Mevcut Sınırlar ve Dikkat Edilmesi Gerekenler

- Authentication/authorization katmanı yoktur.
- `app/core/security.py` boştur.
- Migration sistemi yoktur; tablolar startup sırasında `create_all` ile oluşturulur.
- Celery task dispatch upload sırasında otomatik yapılır, fakat extraction ve normalization manuel endpoint çağrısıyla ilerler.
- `AI_STORE_ANALYSES` ayarı tanımlı olsa da `/analyze-and-save` endpointinde koşul olarak kullanılmaz; endpoint çağrılırsa kayıt yazar.
- Upload edilen dosyalar lokal filesystem storage’a yazılır; S3 veya benzeri harici object storage entegrasyonu yoktur.
- OCR kalitesi Tesseract kurulumuna ve dil paketlerine bağlıdır.
- Embedding modeli ilk kullanımda yüklenir; ortamda model bulunmuyorsa download veya load süreci gecikebilir ya da başarısız olabilir.
- LLM entegrasyonu Ollama servisinin erişilebilir olmasına bağlıdır.
- Forecast için Transformer modeli request sırasında küçük veri üzerinde eğitilir; veri azsa fallback moving average kullanılır.
- Test kapsamı şu an tüm pipeline’ı uçtan uca doğrulamamaktadır.

## 26. Kısa Mimari Özet

Sistem, finansal belgeyi aşamalı biçimde işleyen bir backend motorudur. İlk aşamada dosya yüklenir ve türü belirlenir. İkinci aşamada belge extraction’a hazırlanır; real PDF için orijinal PDF korunurken scanned PDF ve görseller için OCR’a uygun varyantlar üretilir. Üçüncü aşamada native PDF text veya Tesseract OCR ile transaction satırları çıkarılır. Dördüncü aşamada bu ham satırlar normalize edilir, duplicate kayıtlar temizlenir ve kalite skorları hesaplanır. Beşinci aşamada normalize edilmiş işlemler kategorilere ayrılır, harcama profili çıkarılır, anomaliler belirlenir, gelecek dönem harcaması tahmin edilir ve istenirse taksit önerisi ile LLM destekli açıklamalar üretilir.

Bu tasarım, deterministic kuralları ve istatistiksel analizleri merkeze alır; LLM ve embedding modellerini yardımcı/fallback katmanı olarak kullanır. Böylece LLM kapalıyken bile temel analiz fonksiyonları çalışabilir.
