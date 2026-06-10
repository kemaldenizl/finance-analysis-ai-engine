# AI Analiz Motoru — Kalite İyileştirme Planı

> **Amaç:** `/v1/ai/analyze` ve `/v1/ai/analyze-and-save` uçlarının (ve bağlı tüm
> servislerin) ürettiği analizlerin kalitesini artırmak.

## Altın Kurallar (Değiştirilemez Kısıtlar)

1. **Request ve Response JSON modelleri DEĞİŞMEYECEK.** Bu sistemi tüketen
   istemciler (client'lar) mevcut sözleşmeye (contract) göre çalışıyor. Onlara
   ekstra entegrasyon yükü bindirmemek için `app/schemas/analyze.py` içindeki
   alanların adları, tipleri ve yapısı korunacak. Yeni alan **eklenmeyecek**,
   mevcut alan **kaldırılmayacak / yeniden adlandırılmayacak**.
   - Not: `method`, `explanation_method`, `generation_method` gibi alanların
     *değeri* serbest metindir; bunların içeriğini değiştirmek şema değişikliği
     **değildir**, dolayısıyla serbesttir.
2. **LLM değişmeyecek.** Mevcut model (`qwen2.5:1.5b`) aynı kalacak. Bu raporda
   model yükseltme / değiştirme önerisi **yer almaz**. Tüm kazanımlar mevcut
   model, mevcut donanım ve mevcut JSON sözleşmesi sabitken elde edilecektir.

---

## 1. Mevcut Mimari Özeti

Analiz hattı **hibrit** bir yapıdadır: deterministik kural + klasik ML + embedding
+ küçük LLM. LLM yalnızca birkaç noktada **anlatı (narrative)** üretimi ve
**kategori fallback** için devreye girer; çekirdek sinyaller deterministik
üretilir.

| Aşama | Dosya | LLM kullanımı | Çıktı kalitesini belirleyen ana faktör |
|---|---|---|---|
| Özellik çıkarımı | `app/services/ai/feature_engineering.py` | Yok | Veri temizliği, tarih/merchant türetme |
| Kategorizasyon | `app/services/ai/categorization.py` | Sadece fallback | Kural/pattern + embedding kalitesi |
| Profil çıkarımı | `app/services/ai/profiling.py` | Yok | Kategori doğruluğu + eşikler |
| Anomali tespiti | `app/services/ai/anomaly_detection.py` | Sadece açıklama metni | PyOD / robust istatistik + eşikler |
| Tahmin (forecast) | `app/services/ai/forecast_installment.py` | Yok | Veri uzunluğu + yöntem seçimi |
| Taksit önerisi | `app/services/ai/forecast_installment.py` | Sadece açıklama metni | Eşikler + tahmin kalitesi |
| Rapor / Sohbet | `app/services/ai/llm_report.py` | Anlatı + cevap | Prompt + bağlam kalitesi |
| LLM çağrısı | `app/services/ai/providers/ollama_provider.py` | Doğrudan | Ollama parametreleri + prompt |

**Kritik gözlem:** Çıktının kullanıcıya en görünür kısmı (kategori dağılımı,
profil etiketleri, anomali listesi) **LLM'den değil**, deterministik
kural + embedding katmanından gelir. Bu yüzden **en yüksek kalite kazancı
LLM'e dokunmadan**, bu katmanları güçlendirerek elde edilir.

---

## 2. İyileştirme Kalemleri (Önceliklendirilmiş)

Öncelik sırası: **Etki / Maliyet** oranına göre. P0 = en yüksek getiri, en düşük risk.

### P0 — Kategori Taksonomisini Genişletmek (en yüksek etki)

**Dosya:** `app/services/ai/resources/category_taxonomy.yaml`

**Sorun:** Taksonomi çok dar. Şu an yalnızca ~8 kategori ve her birinde çok az
merchant/pattern var. Türk banka/kredi kartı ekstrelerinde sık geçen pek çok
merchant tanımsız. Tanımsız işlem → embedding'e, o da tutmazsa küçük LLM'e
düşüyor; LLM de 1.5B olduğu için çoğu zaman `other` üretiyor. Sonuç: çok sayıda
işlem `other/uncategorized` olarak işaretleniyor ve bu **profil, anomali ve özet**
çıktısını da bozuyor.

**Yapılacak:** YAML'a yaygın Türkiye merchant grupları eklensin (kategori
ID'leri değişebilir; bu **JSON şeması değil**, içerik):
- **Market / gıda:** Migros, BIM, A101, ŞOK, CarrefourSA, Macrocenter, Metro,
  File, Hakmar
- **Akaryakıt:** Shell, BP, Opet, Petrol Ofisi, Total, Aytemiz, Lukoil
- **Ulaşım:** Uber, BiTaksi, Martı, İETT, Metro/Metrobüs, HGS, OGS, otopark
- **Sağlık / eczane:** eczane, hastane, Acıbadem, MLP, Medical Park, lab
- **Giyim:** LC Waikiki, Zara, H&M, Koton, Defacto, Mavi, Boyner
- **Telekom / dijital:** Turkcell, Vodafone, Türk Telekom, Netflix, Spotify,
  YouTube Premium, iCloud, Google
- **Finans:** ATM nakit çekme, havale/EFT, kredi kartı/komisyon, faiz
- **Eğitim, sigorta, abonelik** gibi sık görülen kalemler

**Beklenen kazanım:** `uncategorized_count` belirgin düşer, `rule_assisted_count`
ve `embedding_assisted_count` artar, küçük LLM'e olan bağımlılık azalır
(daha hızlı + daha tutarlı). Profil/anomali/özet de doğal olarak iyileşir.

**Şema etkisi:** Yok. (Sadece çıktıdaki `category` değerleri zenginleşir.)

---

### P0 — Merchant Metnini Eşleştirme Öncesi Normalleştirmek

**Dosya:** `app/services/ai/categorization.py` (`_predict_by_rule`,
`categorize` içindeki `merchant_text` üretimi) ve/veya
`app/services/ai/feature_engineering.py` (`_merchant_name`).

**Sorun:** Eşleştirme ham metin üzerinde yapılıyor. Ekstrelerde merchant adı
büyük/küçük harf, Türkçe karakter (İ/ı, Ş, Ğ, Ü, Ö, Ç), kart ön ekleri
("POS ", "*", şehir/şube kodu, tarih kuyrukları) ile kirli geliyor. Bu da
regex/pattern eşleşmesini düşürüyor.

**Yapılacak:** Eşleştirmeden önce hafif bir normalizasyon katmanı:
- Büyük harfe / küçük harfe sabitleme (tutarlılık)
- Türkçe karakter katlama (folding) — opsiyonel, hem orijinal hem foldlanmış
  metinde arama
- Gürültü token'larını temizleme (kart/POS ön ekleri, ardışık boşluklar,
  sayısal kuyruklar)

**Beklenen kazanım:** Kural ve embedding isabet oranı artar → LLM fallback'i
azalır → daha tutarlı kategori.

**Şema etkisi:** Yok.

---

### P1 — Ollama Çağrı Parametrelerini İyileştirmek

**Dosya:** `app/services/ai/providers/ollama_provider.py`

**Sorunlar ve yapılacaklar:**

1. **`num_ctx` ayarlanmamış.** Ollama varsayılan bağlam penceresi küçük
   (genelde 2048). `generate_structured` içinde prompt'a **hem** JSON schema
   metni ekleniyor **hem de** `format` parametresiyle schema gönderiliyor;
   bu, küçük bağlamı doldurup asıl analiz verisini taşırabiliyor.
   → `options.num_ctx` açıkça yükseltilsin (ör. 4096–8192).
2. **Schema iki kez gönderiliyor.** `generate_structured` zaten `format=schema`
   ile yapısal çıktıyı zorluyor; ayrıca `grounded_prompt` içine schema JSON'unu
   metin olarak gömmek 1.5B model için bağlamı şişiriyor.
   → Prompt'a gömülen schema metni kısaltılsın/kaldırılsın; `format` yeterli.
3. **`seed` yok → tekrar üretilebilirlik (reproducibility) zayıf.**
   → `options.seed` sabitlensin (deterministik çıktı; aynı girdi → aynı sonuç).
4. **Örnekleme parametreleri varsayılan.**
   → `top_p`, `top_k`, `repeat_penalty` finans/özet üretimi için sıkılaştırılsın
   (halüsinasyon ve tekrar riski düşer). `temperature` zaten 0.1, korunabilir.
5. **`num_predict` sınırı yok.**
   → Anlatı çıktıları için makul bir üst sınır → daha hızlı, daha odaklı yanıt.

**Beklenen kazanım:** Daha az bozuk/eksik LLM yanıtı, daha az fallback'e düşme,
daha hızlı ve tekrar üretilebilir çıktı.

**Şema etkisi:** Yok.

---

### P1 — Yapısal LLM Üretimine Yeniden Deneme (Retry) Eklemek

**Dosya:** `app/services/ai/providers/ollama_provider.py`
(`generate_structured`)

**Sorun:** 1.5B model sık sık şemaya uymayan/eksik JSON üretiyor. Şu an tek
denemede `ValidationError` olursa sessizce `None` dönüyor ve çağıran servis
deterministik şablona düşüyor. Bu yüzden anlatı/sohbet cevaplarının önemli
kısmı aslında LLM değil, **şablon** çıktısı oluyor.

**Yapılacak:** 1–2 kez sınırlı **retry** (gerekirse "önceki çıktı geçersizdi,
yalnızca şemaya uygun JSON döndür" şeklinde kısa bir onarım mesajıyla).

**Beklenen kazanım:** Geçerli LLM yanıtı oranı yükselir → anlatı kalitesi artar;
fallback'e düşüş azalır. Başarısızlıkta yine güvenli deterministik çıktı korunur.

**Şema etkisi:** Yok.

---

### P1 — Prompt Kalitesi ve Bağlam Sadeleştirme (Few-shot + Doğal Dil)

**Dosyalar:** `app/services/ai/llm_report.py`,
`app/services/ai/categorization.py` (`_predict_by_llm`)

**Sorunlar ve yapılacaklar:**

1. **Kategorizasyon fallback'inde ipucu yok.** LLM'e yalnızca izin verilen
   kategoriler + ham merchant veriliyor.
   → Embedding'in en yakın 1–2 adayını **ipucu** olarak prompt'a ekle; küçük
   model için birkaç **few-shot örnek** ver (merchant → doğru kategori).
2. **Sohbet bağlamı ham `model_dump()` JSON olarak veriliyor**
   (`answer_question` içindeki `compact_context`). 1.5B model büyük iç içe JSON'u
   iyi yorumlayamıyor.
   → Bağlamı **kısa, doğal dilde maddelenmiş** gerçeklere dönüştür (örn.
   "En yüksek kategori: market %42; tahmini gelecek ay: 12.500 TL; 2 anomali").
3. **Sistem promptları zaten "sayı uydurma" diyor** — bu iyi; korunmalı ve
   few-shot örneklerle pekiştirilmeli.

**Beklenen kazanım:** Daha isabetli kategori fallback'i; daha akıcı ve bağlama
sadık sohbet/özet metni.

**Şema etkisi:** Yok. (Yalnızca `answer`, `text`, `explanation` gibi serbest
metin alanlarının **içeriği** iyileşir.)

---

### P1 — Embedding Eşiği ve Embedding Modeli (LLM'den Bağımsız)

**Dosyalar:** `app/core/config.py` (`EMBEDDING_SIMILARITY_THRESHOLD`,
`EMBEDDING_MODEL_NAME`), `app/services/ai/embedding_classifier.py`

**Not:** Embedding modeli, kural #2'deki **LLM değildir**; ayrı bir
sentence-transformer modelidir. Dolayısıyla bu kalem altın kurallarla çelişmez.

**Sorunlar ve yapılacaklar:**
1. **Eşik (`0.52`) sabit ve elde ayarlanmamış.** Gerçek ekstre verisiyle
   küçük bir doğrulama seti üzerinde eşik taranıp (ör. 0.45–0.60) en iyi
   precision/recall dengesi seçilsin.
2. **Embedding referans dokümanları taksonomiyle birlikte zenginleşir.**
   P0'daki taksonomi genişletmesi embedding isabetini de doğrudan artırır
   (daha çok `examples` → daha iyi benzerlik).
3. (Opsiyonel) Daha güçlü çok dilli bir embedding modeli denenebilir; ancak bu
   donanım/performans dengesi gerektirir, doğrulama seti ile ölçülmeli.

**Beklenen kazanım:** Embedding katmanı daha çok işlemi doğru çözer → LLM
fallback'i azalır → tutarlılık artar.

**Şema etkisi:** Yok. (Mevcut `embedding_model` alanı zaten string; değeri
değişebilir.)

---

### P2 — Tahmin (Forecast) Katmanının Sağlamlaştırılması

**Dosya:** `app/services/ai/forecast_installment.py`

**Sorun:** `MonthlySpendTransformer` her istekte **sıfırdan** ve çok az veri
(`FORECAST_MIN_MONTHS_TRANSFORMER = 6` ay) üzerinde 120 epoch eğitiliyor.
6–12 veri noktasında bir Transformer istatistiksel olarak **aşırı uyum
(overfit)** yapar; sonuç gürültüye yakın ve istekten isteğe oynak olabilir.
Ayrıca her istekte eğitim → gereksiz gecikme.

**Yapılacaklar (şema sabitken):**
- Transformer eşiği yükseltilsin (ör. yeterli veri yoksa daha sağlam yöntemler).
- Az veri rejiminde **mevsimsel naive / Holt-Winters / ağırlıklı hareketli
  ortalama / medyan** gibi sağlam yöntemler tercih edilsin (mevcut
  `moving_average_fallback_v1` zaten var; kapsamı genişletilebilir).
- `confidence` üretimi gerçek hata payına göre kalibre edilsin.
- Tekrar üretilebilirlik için seed zaten sabit (`FORECAST_RANDOM_SEED`); korunur.

**Beklenen kazanım:** Daha kararlı, daha az oynak tahmin → taksit önerisi de
daha güvenilir. Yan fayda: daha hızlı yanıt.

**Şema etkisi:** Yok. (`method` değeri değişir, alan yapısı aynı.)

---

### P2 — Anomali Eşiklerinin Kalibrasyonu

**Dosya:** `app/services/ai/anomaly_detection.py`, `app/core/config.py`
(`ANOMALY_MIN_ROWS_FOR_PYOD`, `ANOMALY_CONTAMINATION`)

**Sorun:** Skor eşikleri (`0.35`, `0.22`), flag bonusları ve `contamination`
elle seçilmiş. Yanlış pozitif/negatif dengesi veriyle doğrulanmamış.

**Yapılacak:** Gerçek ekstrelerden küçük bir etiketli set ile eşik/bonus
değerleri kalibre edilsin; çok az veri rejiminde fallback davranışı gözden
geçirilsin.

**Beklenen kazanım:** Daha anlamlı anomali listesi (gürültü azalır).

**Şema etkisi:** Yok.

---

### P3 — Kalite Skoru ve Durum (status) Eşiklerinin Gözden Geçirilmesi

**Dosya:** `app/services/ai/analysis_service.py` (`_build_quality`,
`status` belirleme: `analysis_confidence < 0.55` → `partial`)

**Yapılacak:** Penalti katsayıları (`0.20`, `0.35`) ve `partial` eşiği gerçek
veriyle gözden geçirilsin; düşük güvenli/invalid işlem oranıyla tutarlı olsun.

**Şema etkisi:** Yok. (`analysis_confidence` ve `status` alanları aynı kalır,
yalnızca hesaplanış kalibre edilir.)

---

## 3. Önerilen Uygulama Sırası (Yol Haritası)

1. **P0 — Taksonomi genişletme** (en yüksek görünür kazanç, sıfır şema riski)
2. **P0 — Merchant normalizasyonu** (taksonomi kazancını çoğaltır)
3. **P1 — Ollama parametreleri** (`num_ctx`, `seed`, schema tekrarını kaldırma)
4. **P1 — Structured generation retry**
5. **P1 — Prompt few-shot + sohbet bağlamını doğal dile çevirme**
6. **P1 — Embedding eşiği kalibrasyonu**
7. **P2 — Forecast sağlamlaştırma**
8. **P2 — Anomali kalibrasyonu**
9. **P3 — Kalite skoru/eşik gözden geçirme**

---

## 4. Doğrulama ve Geri Dönüş (Rollback) Stratejisi

- **Regresyon koruması:** Birkaç gerçek ekstreden oluşan küçük bir "altın set"
  (golden set) oluşturulup, her değişiklik öncesi/sonrası aynı girdilerle
  `analyze` çağrılıp çıktılar karşılaştırılmalı.
- **Sözleşme testi:** Response'un `AnalyzeResponse` şemasına birebir uyduğunu
  doğrulayan bir test korunmalı (alan adı/tip değişmediğinin garantisi).
- **Kademeli devreye alma:** Her kalem ayrı ayrı uygulanıp ölçülmeli; metrikler
  (`uncategorized_count`, `*_assisted_count`, `analysis_confidence`, anomali
  sayısı) izlenmeli.
- **Geri alma:** Taksonomi/eşik/parametre değişiklikleri konfig ve veri
  dosyalarında olduğundan geri alınması kolaydır.

---

## 5. Kapsam Dışı (Bu Planda Yok)

- **LLM yükseltme / değiştirme:** Kural #2 gereği kapsam dışı. Model
  `qwen2.5:1.5b` olarak kalır.
- **Request/Response JSON şema değişiklikleri:** Kural #1 gereği yasak. Yeni
  alan eklenmez, mevcut alan kaldırılmaz/yeniden adlandırılmaz.
