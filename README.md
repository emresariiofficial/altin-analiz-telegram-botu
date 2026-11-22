# 🪙 Dönemsel Gram Altın Analiz ve Otomatik Telegram Bildirim Sistemi

**Beykoz Üniversitesi — Python ve Yapay Zeka Final Projesi**

Bu proje; canlı finansal REST API kaynaklarından veri çeken, **Pandas** ile zaman serisi ve dönemsel fiyat kıyaslaması yapan, **Matplotlib** ile filigranlı analiz grafiği üreten, **Google Gemini / Gemma LLM API** ile finansal yorum çıkaran ve tüm çıktıları **Telegram Bot API** üzerinden otomatik bildirim olarak ileten uçtan uca bir veri boru hattı (data pipeline) mimarisidir.

---

## 🏗️ Sistem Mimarisi (4 Adımlı Veri Boru Hattı)

1. **Canlı Veri Çekme (`requests` + REST API):**
   - Yahoo Finance REST API endpoint'leri üzerinden son 2 yıla ait günlük **Ons Altın (`GC=F`)** ve **Dolar/TL (`TRY=X`)** verileri dinamik olarak JSON formatında çekilir.
2. **Veri İşleme ve Görselleştirme (`pandas` + `matplotlib`):**
   - İki veri seti tarih bazında birleştirilerek günlük **Gram Altın (TL)** fiyatı hesaplanır (`(Ons * Dolar/TL) / 31.1035`).
   - **2025 ve 2026** yıllarının aylık ortalama fiyatları, yıllık fiyat farkları (`TL`), yıllık değişim yüzdeleri (`%`) ve 2026 yılı içi aylık değişim oranları hesaplanır.
   - Sağ alt köşesinde proje filigranı bulunan yüksek çözünürlüklü kıyaslama grafiği (`altin_analiz_raporu.png`) oluşturulur.
3. **Yapay Zeka Finansal Analizi (`google-genai` LLM API):**
   - Pandas ile özetlenen kıyaslama tablosu yapılandırılmış bir istem (prompt) ile LLM modeline aktarılır.
   - API yoğunluğu (HTTP 503) durumlarına karşı otomatik yedek model geçişi (`gemma-4-31b-it` $\rightarrow$ `gemini-3.5-flash` $\rightarrow$ `gemini-3.1-flash-lite`) uygulanmıştır.
4. **Otomatik Bildirim (`Telegram Bot API`):**
   - Üretilen filigranlı grafik ve yapay zeka yatırım analizi raporu Telegram botu aracılığıyla hedef sohbete otomatik gönderilir.

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone <GITHUB_REPO_LINKINIZ>
cd <PROJE_KLASORU>