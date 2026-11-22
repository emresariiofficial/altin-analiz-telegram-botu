import os
import json
import time
from typing import Optional
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env dosyasındaki gizli API anahtarlarını yükle
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GRUP_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Otomatik piyasa değişim eşiği (TL cinsinden) ve kontrol periyodu
FIYAT_DEGISIM_ESIGI_TL = 20.0
KONTROL_ARALIGI_SN = 45

# Kullanıcıların kurduğu özel hedef fiyat alarmları listesi
KULLANICI_ALARMLARI = []

# Türkiye Saati (UTC+3) yardımcısı
TR_SAAT_DILIMI = timezone(timedelta(hours=3))


def tr_simdi() -> datetime:
    return datetime.now(TR_SAAT_DILIMI)


# Gemini istemcisini merkezi olarak başlat
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=25000)
)


def gemini_metin_uret(prompt: str, json_modu: bool = False) -> str:
    """Kotası en yüksek modeller öncelikli, fallback destekli merkezi Gemini çağrı fonksiyonu."""
    adaylar = [
        "gemini-3.1-flash-lite",
        "gemma-4-31b-it",
        "gemini-3.6-flash",
        "gemini-3.8-flash"
    ]
    konfig = types.GenerateContentConfig(response_mime_type="application/json") if json_modu else None

    for model_adi in adaylar:
        try:
            response = gemini_client.models.generate_content(
                model=model_adi,
                contents=prompt,
                config=konfig
            )
            return response.text
        except Exception as e:
            print(f"   ⚠️ {model_adi} atlandı ({str(e)[:50]}...)")

    raise RuntimeError("Tüm Gemini modelleri şu an yoğun.")


def yapay_zeka_niyet_analizi(kullanici_mesaji: str, gonderen_adi: str) -> dict:
    """Gelen her mesajı Gemini'ye inceletir; niyeti, alarm hedefini, ML/Korelasyon veya grafik isteğini JSON döndürür."""
    bugun_str = tr_simdi().strftime("%Y-%m-%d")
    prompt = f"""
    Bugünün tarihi: {bugun_str}.
    Sen Telegram'da çalışan 'Yummi' adında samimi, zeki ve profesyonel bir Altın Analiz Asistanısın.
    Kullanıcı ({gonderen_adi}) sana şu mesajı yazdı: "{kullanici_mesaji}"

    Bu mesajı incele ve SADECE aşağıdaki şemaya uygun geçerli bir JSON objesi döndür:
    {{
        "niyet": "ANLIK_KUR" | "GUNLUK_GRAFIK" | "TARIH_FIYAT" | "GRAFIK_ANALIZ" | "KORELASYON_ANALIZ" | "ML_TAHMIN" | "ALARM_KUR" | "KAYNAK" | "SOHBET",
        "hedef_fiyat": <Eğer niyet ALARM_KUR ise kullanıcının belirttiği Gram Altın (TL) hedef rakamı (örn: 6500.0), değilse null>,
        "hedef_tarih": "<Eğer niyet TARIH_FIYAT ise YYYY-MM-DD formatında kesin tarih (örn: '2025-06-30'), değilse null>",
        "belirli_yil": <Eğer niyet GRAFIK_ANALIZ ise ve '2023 yılının grafiği' gibi TEK BİR TAKVİM YILI istendiyse o yıl (örn: 2023), değilse null>,
        "yil_sayisi": <Eğer niyet GRAFIK_ANALIZ ise ve belirli_yil null ise 'son 5 yıl' için tam sayı (örn: 5), 'max/tüm yıllar' için null, belirtilmediyse 2>,
        "hedef": "DM" | "GRUP" | "AYNI",
        "sohbet_cevabi": "<Eğer niyet SOHBET ise {gonderen_adi} kullanıcısına Yummi karakteriyle vereceğin samimi, kısa ve yardımcı Türkçe cevap; değilse boş string>"
    }}

    Kurallar:
    - 'korelasyon', 'ısı haritası', 'heatmap', 'seaborn', 'veri bilimi analizi', 'ons mu dolar mı daha etkili' isteklerinde niyet = "KORELASYON_ANALIZ".
    - 'tahmin', 'gelecek ay ne olur', 'yapay zeka tahmini', 'makine öğrenmesi', 'regresyon', 'önümüzdeki ay' isteklerinde niyet = "ML_TAHMIN".
    - 'altın 6500 TL olursa haber ver', '6200'e düşerse uyar', 'alarm kur' gibi hedef fiyat bildirim isteklerinde niyet = "ALARM_KUR" ve hedef_fiyat = sayı (float).
    - 'bugünün güncel kurunun grafiği', 'bugünkü altın grafiği', 'gün içi grafik', 'anlık fiyat grafiği' gibi BUGÜNÜN SAATLİK grafiği isteniyorsa niyet = "GUNLUK_GRAFIK".
    - 'bugün altın ne kadar', 'güncel kur' gibi sadece metin olarak anlık fiyat sorularında niyet = "ANLIK_KUR".
    - '30.06.2025 de altın ne kadardı', 'geçen sene 30 haziranda neydi' gibi tek bir günün fiyatında (grafik kelimesi geçmiyorsa) niyet = "TARIH_FIYAT" ve hedef_tarih = "YYYY-MM-DD".
    - '2023 yılının altın grafiği', '2021 grafiğini at' gibi belirli bir yılın grafiği isteniyorsa niyet = "GRAFIK_ANALIZ" ve belirli_yil = 2023.
    - 'son 5 yılın grafiği', 'tüm yılları kıyasla' gibi çok yıllı isteklerde niyet = "GRAFIK_ANALIZ", belirli_yil = null ve yil_sayisi = 5 (veya max için null).
    - 'kaynağın ne', 'veriler nereden' sorularında niyet = "KAYNAK".
    - 'teşekkürler', 'sağ ol', 'merhaba', 'nasılsın' veya genel sohbetlerde niyet = "SOHBET".
    - Mesajda 'bana at', 'at bana', 'bana gönder', 'gönder bana', 'özelden', 'dm', 'şahsıma' geçiyorsa hedef = "DM"; 'gruba' geçiyorsa "GRUP"; aksi halde "AYNI".
    """
    try:
        ham_yanit = gemini_metin_uret(prompt, json_modu=True)
        temiz_json = ham_yanit.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(temiz_json)
    except Exception as e:
        print(f"⚠️ Niyet analizi yedek moda düştü: {e}")
        return {
            "niyet": "SOHBET",
            "hedef_fiyat": None,
            "hedef_tarih": None,
            "belirli_yil": None,
            "yil_sayisi": 2,
            "hedef": "AYNI",
            "sohbet_cevabi": f"Rica ederim {gonderen_adi}! Ben Yummi, her zaman buradayım. Güncel altın kurunu, korelasyon ısı haritasını veya yapay zeka fiyat tahminini isteyebilirsin."
        }


def canli_fiyat_verisi_al() -> tuple:
    """Yahoo Finance API'den anlık (gram_altin_tl, ons_usd, dolar_tl) üçlüsünü döndürür."""
    headers = {"User-Agent": "Mozilla/5.0"}
    res_ons = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=1d&interval=1m", headers=headers, timeout=10).json()
    res_kur = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/TRY=X?range=1d&interval=1m", headers=headers, timeout=10).json()

    ons_fiyat = float(res_ons["chart"]["result"][0]["meta"]["regularMarketPrice"])
    kur_fiyat = float(res_kur["chart"]["result"][0]["meta"]["regularMarketPrice"])
    gram_altin = (ons_fiyat * kur_fiyat) / 31.1035
    return gram_altin, ons_fiyat, kur_fiyat


def finans_verisi_cek(sembol: str, aralik: str = "2y", periyot: str = "1d") -> pd.DataFrame:
    """Doğrudan REST API'den veriyi çeker, Pandas ile eksik ve mükerrer kayıtları temizler."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sembol}?range={aralik}&interval={periyot}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    veri = response.json()

    sonuc = veri["chart"]["result"][0]
    zamanlar = sonuc["timestamp"]
    kapanislar = sonuc["indicators"]["quote"][0]["close"]

    zaman_serisi = pd.to_datetime(zamanlar, unit="s")
    if periyot != "1d":
        zaman_serisi = zaman_serisi.tz_localize("UTC").tz_convert("Europe/Istanbul")

    df = pd.DataFrame({
        "Zaman": zaman_serisi if periyot != "1d" else zaman_serisi.date,
        sembol: kapanislar
    })
    # Veri temizleme aşaması: kirli/eksik ve tekrar eden satırları silme
    return df.dropna().drop_duplicates(subset=["Zaman"])


def korelasyon_ve_barplot_analizi_gonder(hedef_chat_id: str):
    """Seaborn ile Korelasyon Isı Haritası (Heatmap) ve Aylık Ortalama Barplot çizer, istatistiksel yorumla gönderir."""
    print("[Veri Bilimi] 5 yıllık veri çekiliyor, Seaborn Heatmap ve Barplot oluşturuluyor...")
    df_ons = finans_verisi_cek("GC=F", aralik="5y", periyot="1d")
    df_kur = finans_verisi_cek("TRY=X", aralik="5y", periyot="1d")

    df = pd.merge(df_ons, df_kur, on="Zaman", how="inner").dropna().drop_duplicates()
    df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035
    df["Tarih"] = pd.to_datetime(df["Zaman"])
    df["Ay_No"] = df["Tarih"].dt.month

    ay_isimleri = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }
    df["Ay"] = df["Ay_No"].map(ay_isimleri)

    # Korelasyon matrisi için sayısal kolonları seç
    korelasyon_df = df[["GC=F", "TRY=X", "Gram_Altin_TL"]].rename(columns={
        "GC=F": "Ons Altın ($)",
        "TRY=X": "Dolar / TL",
        "Gram_Altin_TL": "Gram Altın (TL)"
    })
    korelasyon_matrisi = korelasyon_df.corr()

    # Aylık ortalama performans sıralaması (Barplot için)
    aylik_ort = df.groupby("Ay", as_index=False)["Gram_Altin_TL"].mean().sort_values(by="Gram_Altin_TL", ascending=False)

    grafik_yolu = "korelasyon_ve_dagilim.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    # Sol Grafik: Seaborn Korelasyon Isı Haritası (Heatmap)
    sns.heatmap(korelasyon_matrisi, annot=True, fmt=".2f", cmap="YlOrBr", ax=axes[0], cbar=True)
    axes[0].set_title("Değişkenler Arası Korelasyon Isı Haritası", fontsize=12, fontweight="bold")

    # Sağ Grafik: Seaborn Aylık Ortalama Barplot
    sns.barplot(data=aylik_ort, x="Gram_Altin_TL", y="Ay", palette="viridis", hue="Ay", legend=False, ax=axes[1])
    axes[1].set_title("Son 5 Yılın Aylık Ortalama Gram Altın Sıralaması", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ortalama Gram Altın (TL)")
    axes[1].set_ylabel("Aylar")

    fig.suptitle("Veri Bilimi & Keşifçi Veri Analizi (EDA) Raporu", fontsize=14, fontweight="bold")
    plt.figtext(0.99, 0.01, "Beykoz Üni - Python & Yapay Zeka Final Projesi", fontsize=9, color="gray", ha="right")
    plt.tight_layout()
    plt.savefig(grafik_yolu, dpi=300)
    plt.close()

    ons_corr = korelasyon_matrisi.loc["Ons Altın ($)", "Gram Altın (TL)"]
    kur_corr = korelasyon_matrisi.loc["Dolar / TL", "Gram Altın (TL)"]
    en_yuksek_ay = aylik_ort.iloc[0]["Ay"]
    en_yuksek_deger = aylik_ort.iloc[0]["Gram_Altin_TL"]

    aciklama = (
        f"🔬 VERİ BİLİMİ & KORELASYON ANALİZİ (Seaborn EDA)\n\n"
        f"• Dolar/TL ↔ Gram Altın Korelasyonu: {kur_corr:.2f}\n"
        f"• Ons Altın ↔ Gram Altın Korelasyonu: {ons_corr:.2f}\n"
        f"• Tarihsel En Yüksek Ortalama Ay: {en_yuksek_ay} ({en_yuksek_deger:,.2f} ₺)\n\n"
        f"📌 İstatistiksel Çıkarım: Korelasyon katsayılarına göre Türkiye'de Gram Altın fiyatı üzerinde "
        f"{'Dolar/TL kuru' if kur_corr >= ons_corr else 'Küresel Ons Altın fiyatı'} daha belirleyici bir etkiye sahiptir."
    )

    with open(grafik_yolu, "rb") as foto:
        telegram_istek_gonder(
            "sendPhoto",
            data={"chat_id": hedef_chat_id, "caption": "🔬 Korelasyon Isı Haritası (Heatmap) ve Aylık Dağılım Grafiği"},
            files={"photo": foto}
        )
    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_chat_id, "text": aciklama})
    print("✅ Korelasyon ısı haritası ve Seaborn barplot Telegram'a iletildi.")


def makine_ogrenmesi_fiyat_tahmini_gonder(hedef_chat_id: str):
    """Scikit-Learn MinMaxScaler ile veriyi ölçekler ve LinearRegression ile 30 günlük trend tahmini üretir."""
    print("[ML Tahmin] MinMaxScaler ön işleme ve Lineer Regresyon modeli çalıştırılıyor...")
    df_ons = finans_verisi_cek("GC=F", aralik="1y", periyot="1d")
    df_kur = finans_verisi_cek("TRY=X", aralik="1y", periyot="1d")

    df = pd.merge(df_ons, df_kur, on="Zaman", how="inner").dropna().drop_duplicates()
    df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035
    df = df.reset_index(drop=True)
    df["Gun_Indeksi"] = np.arange(len(df))

    # Yapay Zeka Ön İşleme: MinMaxScaler ile 0-1 aralığına normalizasyon
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(df[["Gun_Indeksi"]])
    y_scaled = scaler_y.fit_transform(df[["Gram_Altin_TL"]])

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    # Gelecek 30 işlem günü için tahmin üret
    son_indeks = df["Gun_Indeksi"].iloc[-1]
    gelecek_indeksler = np.arange(son_indeks + 1, son_indeks + 31).reshape(-1, 1)
    gelecek_x_scaled = scaler_x.transform(pd.DataFrame(gelecek_indeksler, columns=["Gun_Indeksi"]))
    tahmin_scaled = model.predict(gelecek_x_scaled)
    tahmin_tl = scaler_y.inverse_transform(tahmin_scaled).flatten()

    mevcut_fiyat = df["Gram_Altin_TL"].iloc[-1]
    otuz_gun_sonra = tahmin_tl[-1]
    beklenen_degisim = ((otuz_gun_sonra - mevcut_fiyat) / mevcut_fiyat) * 100
    r2_skoru = model.score(X_scaled, y_scaled)

    # Grafik çizimi
    grafik_yolu = "ml_fiyat_tahmini.png"
    plt.figure(figsize=(12, 5.5))
    tarihler = pd.to_datetime(df["Zaman"])
    gelecek_tarihler = pd.date_range(start=tarihler.iloc[-1] + pd.Timedelta(days=1), periods=30, freq="B")

    plt.plot(tarihler, df["Gram_Altin_TL"], label="Gerçekleşen Son 1 Yıl (TL)", color="#1f77b4", linewidth=2)
    plt.plot(gelecek_tarihler, tahmin_tl, label="ML Lineer Regresyon Tahmini (30 İş Günü)", color="#d62728", linestyle="--", linewidth=2.5)
    plt.title("Scikit-Learn (MinMaxScaler + Lineer Regresyon) Altın Fiyat Trend Tahmini", fontsize=12.5, fontweight="bold")
    plt.xlabel("Tarih")
    plt.ylabel("Gram Altın (TL)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.text(0.99, 0.02, "Beykoz Üni - Python & Yapay Zeka Final Projesi", fontsize=9, color="gray", ha="right", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(grafik_yolu, dpi=300)
    plt.close()

    mesaj = (
        f"🤖 MAKİNE ÖĞRENMESİ (ML) FİYAT TREND TAHMİNİ\n\n"
        f"• Kullanılan Yöntem: MinMaxScaler (0-1 Normalizasyon) + Lineer Regresyon\n"
        f"• Model Uyum Skoru (R²): {r2_skoru:.2f}\n"
        f"• Güncel Gram Altın: {mevcut_fiyat:,.2f} ₺\n"
        f"• 30 İş Günü Sonrası Trend Hedefi: {otuz_gun_sonra:,.2f} ₺ (%{beklenen_degisim:+.2f})\n\n"
        f"⚠️ Not: Bu çıktı geçmiş 1 yıllık verinin doğrusal regresyon eğilimini gösterir, kesin yatırım tavsiyesi değildir."
    )

    with open(grafik_yolu, "rb") as foto:
        telegram_istek_gonder(
            "sendPhoto",
            data={"chat_id": hedef_chat_id, "caption": "🤖 Makine Öğrenmesi (MinMaxScaler + Lineer Regresyon) Tahmin Grafiği"},
            files={"photo": foto}
        )
    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_chat_id, "text": mesaj})
    print("✅ ML fiyat tahmini grafiği ve raporu Telegram'a iletildi.")


def saatlik_otomatik_kur_mesaji() -> tuple:
    """Her saat başı gruba otomatik atılacak kısa güncel altın fiyat mesajını ve o anki gram fiyatını döndürür."""
    gram_altin, ons_fiyat, kur_fiyat = canli_fiyat_verisi_al()
    simdi_str = tr_simdi().strftime("%d.%m.%Y %H:%M")

    mesaj = (
        f"⏰ SAATLİK KUR BİLDİRİMİ ({simdi_str})\n"
        f"Şu an altının güncel fiyatı: {gram_altin:,.2f} TL\n"
        f"(Ons: {ons_fiyat:,.2f} $ | Dolar: {kur_fiyat:,.4f} ₺)"
    )
    return mesaj, gram_altin


def gunluk_canli_grafik_gonder(hedef_chat_id: str):
    """Bugünün (son 24 saatin) 15 dakikalık canlı Gram Altın (TL) hareket grafiğini çizer ve gönderir."""
    print("[1/3] Bugünün 15 dakikalık canlı Ons Altın ve Dolar/TL verileri çekiliyor...")
    df_ons = finans_verisi_cek("GC=F", aralik="1d", periyot="15m")
    df_kur = finans_verisi_cek("TRY=X", aralik="1d", periyot="15m")

    df = pd.merge_asof(
        df_ons.sort_values("Zaman"),
        df_kur.sort_values("Zaman"),
        on="Zaman",
        direction="nearest"
    )
    df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035
    df["Saat"] = df["Zaman"].dt.strftime("%H:%M")

    guncel_fiyat = df["Gram_Altin_TL"].iloc[-1]
    acilis_fiyat = df["Gram_Altin_TL"].iloc[0]
    en_yuksek = df["Gram_Altin_TL"].max()
    en_dusuk = df["Gram_Altin_TL"].min()
    degisim_yuzde = ((guncel_fiyat - acilis_fiyat) / acilis_fiyat) * 100
    tarih_bugun = tr_simdi().strftime("%d.%m.%Y")

    grafik_yolu = "gunluk_altin_grafigi.png"
    plt.figure(figsize=(12, 5.5))
    plt.plot(df["Saat"], df["Gram_Altin_TL"], marker="o", markersize=4, linewidth=2.2, color="#D4AF37", label="Gram Altın (TL)")
    plt.fill_between(df["Saat"], df["Gram_Altin_TL"], min(df["Gram_Altin_TL"]) * 0.999, color="#D4AF37", alpha=0.15)

    plt.title(f"Bugünün Canlı Gram Altın (TL) Hareket Grafiği ({tarih_bugun})", fontsize=13, fontweight="bold")
    plt.xlabel("Saat (TSİ)")
    plt.ylabel("Gram Altın (TL)")
    plt.xticks(df["Saat"][::max(1, len(df) // 12)], rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.text(
        0.99, 0.02, "Beykoz Üni - Python & Yapay Zeka Final Projesi",
        fontsize=9, color="gray", alpha=0.75,
        ha="right", va="bottom", transform=plt.gca().transAxes
    )
    plt.tight_layout()
    plt.savefig(grafik_yolu, dpi=300)
    plt.close()

    ozet_mesaj = (
        f"📈 BUGÜNÜN CANLI GRAM ALTIN ÖZETİ ({tarih_bugun})\n\n"
        f"• Güncel Fiyat: {guncel_fiyat:,.2f} ₺ (%{degisim_yuzde:+.2f})\n"
        f"• Gün İçi En Yüksek: {en_yuksek:,.2f} ₺\n"
        f"• Gün İçi En Düşük: {en_dusuk:,.2f} ₺\n"
        f"• Son Ons / Dolar: {df['GC=F'].iloc[-1]:,.2f} $ | {df['TRY=X'].iloc[-1]:,.4f} ₺\n\n"
        f"💬 Geçmiş yılların analizini isterseniz '2023 yılının grafiğini at' veya 'korelasyon haritası at' yazabilirsiniz."
    )

    with open(grafik_yolu, "rb") as foto:
        telegram_istek_gonder(
            "sendPhoto",
            data={"chat_id": hedef_chat_id, "caption": f"📊 {tarih_bugun} Gün İçi Canlı Gram Altın (TL) Grafiği"},
            files={"photo": foto}
        )
    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_chat_id, "text": ozet_mesaj})
    print("✅ Günlük canlı grafik ve özet Telegram'a iletildi.")


def gecmis_tarih_fiyat_getir(hedef_tarih_str: str) -> str:
    """Belirli bir geçmiş tarihteki (hafta sonuysa en yakın önceki işlem günündeki) Gram Altın fiyatını getirir."""
    try:
        hedef_dt = datetime.strptime(hedef_tarih_str, "%Y-%m-%d").date()
        if hedef_dt > tr_simdi().date():
            return f"⚠️ {hedef_dt.strftime('%d.%m.%Y')} henüz gelmemiş bir tarih! Geçmiş bir tarih veya güncel kur sorabilirsin."

        fark_yil = tr_simdi().year - hedef_dt.year
        aralik = "1y" if fark_yil <= 1 else ("5y" if fark_yil <= 5 else "max")

        df_ons = finans_verisi_cek("GC=F", aralik=aralik)
        df_kur = finans_verisi_cek("TRY=X", aralik=aralik)
        df = pd.merge(df_ons, df_kur, on="Zaman", how="inner")
        df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035

        uygun_kayitlar = df[df["Zaman"] <= hedef_dt]
        if uygun_kayitlar.empty:
            return f"⚠️ {hedef_dt.strftime('%d.%m.%Y')} tarihi için API arşivinde veri bulunamadı."

        secilen = uygun_kayitlar.iloc[-1]
        gercek_tarih = secilen["Zaman"].strftime("%d.%m.%Y")
        istenen_tarih = hedef_dt.strftime("%d.%m.%Y")
        tatil_notu = f"\nℹ️ ({istenen_tarih} piyasa tatiline denk geldiği için en yakın işlem günü olan {gercek_tarih} verisi alınmıştır.)" if gercek_tarih != istenen_tarih else ""

        return (
            f"📅 TARİHSEL ALTIN KURU SORGUSU ({gercek_tarih})\n\n"
            f"• Gram Altın (TL): {secilen['Gram_Altin_TL']:,.2f} ₺\n"
            f"• Ons Altın (USD): {secilen['GC=F']:,.2f} $\n"
            f"• Dolar / TL: {secilen['TRY=X']:,.4f} ₺"
            f"{tatil_notu}\n\n"
            f"💬 Başka bir tarih veya çok yıllı grafik analizi ister misiniz?"
        )
    except Exception as e:
        return f"⚠️ Belirtilen tarih ({hedef_tarih_str}) sorgulanırken hata oluştu: {e}"


def anlik_altin_kuru_getir() -> str:
    """Canlı API'den o anki güncel Ons Altın, Dolar/TL ve hesaplanan Gram Altın (TL) fiyatını döndürür."""
    gram_altin, ons_fiyat, kur_fiyat = canli_fiyat_verisi_al()
    zaman_damgasi = tr_simdi().strftime("%d.%m.%Y %H:%M")

    return (
        f"💰 GÜNCEL CANLI ALTIN VE DÖVİZ KURLARI ({zaman_damgasi})\n\n"
        f"• Gram Altın (TL): {gram_altin:,.2f} ₺\n"
        f"• Ons Altın (USD): {ons_fiyat:,.2f} $\n"
        f"• Dolar / TL: {kur_fiyat:,.4f} ₺\n\n"
        f"ℹ️ Veriler canlı finans API'si üzerinden anlık hesaplanmıştır.\n\n"
        f"💬 Başka bir isteğiniz var mı? (Örn: 'altın 6500 TL olursa haber ver' veya 'gelecek ay altın tahmini')"
    )


def veri_isle_ve_gorsellestir(yil_sayisi: Optional[int] = 2, belirli_yil: Optional[int] = None) -> tuple:
    guncel_yil = tr_simdi().year

    if belirli_yil is not None:
        fark = max(1, guncel_yil - belirli_yil + 1)
        aralik_str = "max" if fark > 10 else f"{fark}y"
        etiket = f"{belirli_yil} yılı"
    else:
        aralik_str = "max" if yil_sayisi is None else f"{yil_sayisi}y"
        etiket = "TÜM TARİHSEL (MAX)" if yil_sayisi is None else f"son {yil_sayisi} yıllık"

    print(f"[1/4] Canlı API'den {etiket} Ons Altın ve Dolar/TL verileri çekiliyor...")
    df_ons = finans_verisi_cek("GC=F", aralik=aralik_str)
    df_kur = finans_verisi_cek("TRY=X", aralik=aralik_str)

    print(f"[2/4] Pandas ile veriler temizleniyor ve {etiket} analizi yapılıyor...")
    df = pd.merge(df_ons, df_kur, on="Zaman", how="inner")
    df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035
    df["Tarih"] = pd.to_datetime(df["Zaman"])

    df["Yil"] = df["Tarih"].dt.year
    df["Ay"] = df["Tarih"].dt.month

    if belirli_yil is not None:
        df = df[df["Yil"] == belirli_yil]
    elif yil_sayisi is not None:
        hedef_yillar = list(range(guncel_yil - yil_sayisi + 1, guncel_yil + 1))
        df = df[df["Yil"].isin(hedef_yillar)]

    ozet = df.groupby(["Ay", "Yil"])["Gram_Altin_TL"].mean().unstack()
    mevcut_yillar = list(ozet.columns)
    ozet.columns = [f"{y}_Ort_TL" for y in mevcut_yillar]

    ilk_yil_kolon = f"{mevcut_yillar[0]}_Ort_TL"
    son_yil_kolon = f"{mevcut_yillar[-1]}_Ort_TL"

    if len(mevcut_yillar) > 1:
        ozet[f"{mevcut_yillar[0]}_{mevcut_yillar[-1]}_Degisim_%"] = (
            (ozet[son_yil_kolon] - ozet[ilk_yil_kolon]) / ozet[ilk_yil_kolon]
        ) * 100
    ozet[f"{mevcut_yillar[-1]}_Aylik_Degisim_%"] = ozet[son_yil_kolon].pct_change(fill_method=None) * 100

    ay_isimleri = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }
    ozet.index = ozet.index.map(ay_isimleri)
    ozet = ozet.round(2)

    grafik_yolu = "altin_analiz_raporu.png"
    plt.figure(figsize=(13, 6.5))
    isaretciler = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]

    for idx, yil in enumerate(mevcut_yillar):
        kolon = f"{yil}_Ort_TL"
        kalinlik = 3.0 if yil == mevcut_yillar[-1] else 1.5
        plt.plot(
            ozet.index,
            ozet[kolon],
            marker=isaretciler[idx % len(isaretciler)],
            linewidth=kalinlik,
            label=f"{yil}"
        )

    baslik_yillar = f"{mevcut_yillar[0]} - {mevcut_yillar[-1]}" if len(mevcut_yillar) > 1 else f"{mevcut_yillar[0]} Yılı"
    plt.title(f"{baslik_yillar} Dönemi Aylık Ortalama Gram Altın Analizi (TL)", fontsize=13, fontweight="bold")
    plt.xlabel("Aylar")
    plt.ylabel("Gram Altın (TL)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(ncol=3 if len(mevcut_yillar) > 8 else 1, fontsize=8.5)

    plt.text(
        0.99, 0.02, "Beykoz Üni - Python & Yapay Zeka Final Projesi",
        fontsize=9, color="gray", alpha=0.75,
        ha="right", va="bottom", transform=plt.gca().transAxes
    )

    plt.tight_layout()
    plt.savefig(grafik_yolu, dpi=300)
    plt.close()

    return ozet, grafik_yolu, mevcut_yillar


def llm_ile_analiz_et(ozet_df: pd.DataFrame, mevcut_yillar: list) -> str:
    print("[3/4] Tablo verisi Gemini LLM API'ye gönderiliyor...")
    donem_uzunlugu = len(mevcut_yillar)
    tablo_metni = ozet_df.to_string()
    donem_tanimi = f"{mevcut_yillar[0]} yılına ait" if donem_uzunlugu == 1 else f"{mevcut_yillar[0]} ile {mevcut_yillar[-1]} yılları arasındaki ({donem_uzunlugu} yıllık)"

    prompt = f"""
    Sen kıdemli bir emtia ve finans analistisin. Kullanıcı {donem_tanimi} Gram Altın (TL) performansını analiz etmeni istedi.
    Aşağıda canlı API'den çekilmiş aylık ortalama Gram Altın (TL) fiyatları ve değişim oranları yer alıyor:

    {tablo_metni}

    Bu verilere dayanarak yatırımcılar için kısa, net ve profesyonel bir rapor hazırla:
    1. Seçilen dönem boyunca altın fiyatlarının genel trendini ve yıl içinde hangi aylarda gevşeme (düşüş veya yavaşlama) gösterdiğini belirt.
    2. Dönem başındaki fiyat ile dönem sonundaki fiyat arasındaki değişimi rakamlarla özetle.
    3. Yatırımcılar için 3 maddelik somut çıkarım yap.
    (Not: Telegram mesajı olarak gönderileceği için markdown başlıkları yerine sade madde işaretleri ve kalın vurgular kullan.)
    """
    return gemini_metin_uret(prompt, json_modu=False)


def telegram_istek_gonder(metot: str, data: dict, files=None, deneme_sayisi: int = 3):
    """PythonAnywhere proxy kesintilerine (503) karşı otomatik yeniden denemeli Telegram göndericisi."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{metot}"
    for deneme in range(1, deneme_sayisi + 1):
        try:
            if files:
                for f in files.values():
                    f.seek(0)
            res = requests.post(url, data=data, files=files, timeout=15)
            res.raise_for_status()
            return res
        except Exception as e:
            if deneme == deneme_sayisi:
                raise e
            print(f"   🔄 Telegram bağlantısı yenileniyor ({deneme}/{deneme_sayisi})...")
            time.sleep(2)


def telegrama_gonder(rapor_metni: str, grafik_yolu: str, hedef_chat_id: str, mevcut_yillar: list):
    donem_baslik = str(mevcut_yillar[0]) if len(mevcut_yillar) == 1 else f"{mevcut_yillar[0]}-{mevcut_yillar[-1]}"
    print(f"[4/4] Analiz raporu ve grafik hedef sohbete ({hedef_chat_id}) gönderiliyor...")

    with open(grafik_yolu, "rb") as foto:
        telegram_istek_gonder(
            "sendPhoto",
            data={
                "chat_id": hedef_chat_id,
                "caption": f"📊 {donem_baslik} Dönemi Aylık Ortalama Gram Altın Grafiği"
            },
            files={"photo": foto}
        )

    devam_notu = "\n\n💬 Başka bir analiz veya grafik ister misiniz? (Örn: 'korelasyon haritası at' veya 'gelecek ay altın tahmini')"
    baslik = f"🪙 {donem_baslik} ({len(mevcut_yillar)} YILLIK) ALTIN ANALİZ RAPORU\n\n"
    tam_mesaj = (baslik + rapor_metni)[:3850] + devam_notu

    telegram_istek_gonder(
        "sendMessage",
        data={
            "chat_id": hedef_chat_id,
            "text": tam_mesaj
        }
    )
    print("✅ İşlem tamamlandı! Rapor ve grafik Telegram'a başarıyla iletildi.")


def boru_hattini_calistir(yil_sayisi: Optional[int] = 2, belirli_yil: Optional[int] = None, hedef_chat_id: str = GRUP_CHAT_ID):
    ozet_tablo, grafik_dosyasi, mevcut_yillar = veri_isle_ve_gorsellestir(yil_sayisi=yil_sayisi, belirli_yil=belirli_yil)
    print("\n--- PANDAS ÖZET TABLOSU ---")
    print(ozet_tablo)

    analiz_sonucu = llm_ile_analiz_et(ozet_tablo, mevcut_yillar=mevcut_yillar)
    print("\n--- GEMINI ANALİZ ÇIKTISI ---")
    print(analiz_sonucu)

    telegrama_gonder(analiz_sonucu, grafik_dosyasi, hedef_chat_id=hedef_chat_id, mevcut_yillar=mevcut_yillar)


def telegram_bot_dinle():
    """Canlı piyasa takibi, özel hedef fiyat alarmları, Veri Bilimi (Seaborn/ML) ve Gemini AI Niyet Yönlendiricisi."""
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    son_update_id = None

    ilk_yanit = requests.get(f"{base_url}/getUpdates", timeout=15).json()
    if ilk_yanit.get("result"):
        son_update_id = ilk_yanit["result"][-1]["update_id"] + 1

    karsilama_mesaji = (
        "Selam Hoşgeldiniz! Ben Altın Analiz Asistanınız Yummi, size nasıl yardımcı olabilirim?\n"
        "🚨 Altın fiyatlarında ani düşüş veya yükseliş olduğunda ve her saat başı size anında bildireceğim.\n\n"
        "💡 Neler Yapabilirim? (Doğrudan yazmanız yeterli):\n"
        "• güncel altın kuru ne (Anlık Gram, Ons ve Dolar kurunu söyler)\n"
        "• altın 6500 TL olursa haber ver (Belirlediğiniz fiyata ulaşınca özel alarm verir)\n"
        "• bugünün güncel kurunun grafiğini at (Gün içi saatlik canlı grafik çizer)\n"
        "• 2023 yılının altın grafiğini at (Belirli bir yılın grafiğini çizer)\n"
        "• korelasyon ısı haritası at (Ons, Dolar ve Gram ilişkisini Seaborn ile analiz eder)\n"
        "• gelecek ay altın tahmini (Makine Öğrenmesi ile 30 günlük trend tahmini yapar)"
    )

    hedef_grup = str(GRUP_CHAT_ID).strip() if GRUP_CHAT_ID else ""
    print(f"📡 Hedef Grup ID (.env): '{hedef_grup}'")

    if hedef_grup:
        try:
            telegram_istek_gonder("sendMessage", data={"chat_id": hedef_grup, "text": karsilama_mesaji})
            print("✅ Karşılama mesajı gruba başarıyla gönderildi!")
        except Exception as e:
            print(f"❌ Karşılama mesajı gönderilemedi: {e}")

    referans_gram_fiyat = None
    try:
        referans_gram_fiyat, _, _ = canli_fiyat_verisi_al()
        print(f"📌 Başlangıç Referans Gram Altın Fiyatı: {referans_gram_fiyat:,.2f} TL")
    except Exception as e:
        print(f"⚠️ İlk referans fiyat alınamadı, ilk kontrolde güncellenecek: {e}")

    print("\n🐱 Yummi Aktif! Canlı piyasa alarmları, Veri Bilimi modülleri ve AI Router devrede... (Durdurmak için Ctrl+C)\n")

    son_bildirim_saati = tr_simdi().strftime("%Y-%m-%d %H")
    son_fiyat_kontrol_zamani = time.time()

    while True:
        try:
            simdi_tr = tr_simdi()

            # --- 1. CANLI FİYAT KONTROLÜ (Otomatik Ani Değişim + Kullanıcı Özel Alarmları) ---
            if (time.time() - son_fiyat_kontrol_zamani) >= KONTROL_ARALIGI_SN:
                son_fiyat_kontrol_zamani = time.time()
                try:
                    guncel_gram, guncel_ons, guncel_kur = canli_fiyat_verisi_al()
                    simdi_str = simdi_tr.strftime("%d.%m.%Y %H:%M")

                    # A) Ani Piyasa Değişimi Bildirimi
                    if hedef_grup:
                        if referans_gram_fiyat is None:
                            referans_gram_fiyat = guncel_gram
                        else:
                            fark_tl = guncel_gram - referans_gram_fiyat
                            if abs(fark_tl) >= FIYAT_DEGISIM_ESIGI_TL:
                                yon_ikon = "📈 ANİ YÜKSELİŞ BİLDİRİMİ" if fark_tl > 0 else "📉 ANİ DÜŞÜŞ BİLDİRİMİ"
                                alarm_mesaji = (
                                    f"🚨 {yon_ikon} ({simdi_str})\n"
                                    f"Altın fiyatında {fark_tl:+.2f} TL hareket gerçekleşti!\n\n"
                                    f"• Şu Anki Fiyat: {guncel_gram:,.2f} TL\n"
                                    f"• Önceki Seviye: {referans_gram_fiyat:,.2f} TL\n"
                                    f"• Ons / Dolar: {guncel_ons:,.2f} $ | {guncel_kur:,.4f} ₺"
                                )
                                telegram_istek_gonder("sendMessage", data={"chat_id": hedef_grup, "text": alarm_mesaji})
                                print(f"🚨 Ani fiyat hareketi bildirildi: {referans_gram_fiyat:,.2f} -> {guncel_gram:,.2f} TL")
                                referans_gram_fiyat = guncel_gram

                    # B) Kullanıcıların Kurduğu Hedef Fiyat Alarmları
                    for alarm in list(KULLANICI_ALARMLARI):
                        hedef_f = alarm["hedef_fiyat"]
                        tetiklendi = (alarm["yon"] == "YUKARI" and guncel_gram >= hedef_f) or (alarm["yon"] == "ASAGI" and guncel_gram <= hedef_f)
                        if tetiklendi:
                            ozel_alarm_msg = (
                                f"🔔 ÖZEL HEDEF FİYAT ALARMI ({simdi_str})\n"
                                f"Hey {alarm['kisi']}, belirlediğin seviyeye ulaşıldı!\n\n"
                                f"• Hedef Fiyatın: {hedef_f:,.2f} TL\n"
                                f"• Güncel Gram Altın: {guncel_gram:,.2f} TL\n"
                                f"• Ons / Dolar: {guncel_ons:,.2f} $ | {guncel_kur:,.4f} ₺"
                            )
                            telegram_istek_gonder("sendMessage", data={"chat_id": alarm["chat_id"], "text": ozel_alarm_msg})
                            KULLANICI_ALARMLARI.remove(alarm)
                            print(f"🔔 {alarm['kisi']} kullanıcısının {hedef_f:,.2f} TL alarmı tetiklendi!")

                except Exception as alarm_hatasi:
                    print(f"⚠️ Anlık fiyat kontrolü atlandı: {alarm_hatasi}")

            # --- 2. SAATLİK OTOMATİK KUR BİLDİRİMİ ---
            suanki_saat_anahtari = simdi_tr.strftime("%Y-%m-%d %H")
            if hedef_grup and suanki_saat_anahtari != son_bildirim_saati and simdi_tr.minute == 0:
                try:
                    otomatik_mesaj, guncel_gram = saatlik_otomatik_kur_mesaji()
                    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_grup, "text": otomatik_mesaj})
                    son_bildirim_saati = suanki_saat_anahtari
                    referans_gram_fiyat = guncel_gram
                    print(f"⏰ Saatlik kur bildirimi gruba gönderildi ({simdi_tr.strftime('%H:%M')}).")
                except Exception as saat_hatasi:
                    print(f"⚠️ Saatlik bildirim gönderilemedi: {saat_hatasi}")

            # --- 3. GELEN TELEGRAM MESAJLARINI DİNLE ---
            params = {"timeout": 1}
            if son_update_id:
                params["offset"] = son_update_id

            yanit = requests.get(f"{base_url}/getUpdates", params=params, timeout=15).json()
            for guncelleme in yanit.get("result", []):
                son_update_id = guncelleme["update_id"] + 1
                mesaj = guncelleme.get("message") or guncelleme.get("channel_post")
                if not mesaj:
                    continue

                sohbet_id = str(mesaj["chat"]["id"])

                if "new_chat_members" in mesaj:
                    for yeni_uye in mesaj["new_chat_members"]:
                        if not yeni_uye.get("is_bot"):
                            yeni_isim = yeni_uye.get("first_name", "Dostum")
                            hosgeldin = f"👋 Aramıza hoş geldin {yeni_isim}!\n\n{karsilama_mesaji}"
                            telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": hosgeldin})
                    continue

                if "text" not in mesaj or mesaj.get("from", {}).get("is_bot"):
                    continue

                ham_metin = mesaj["text"].strip()
                gonderen_id = str(mesaj["from"]["id"]) if "from" in mesaj else sohbet_id
                gonderen_adi = mesaj.get("from", {}).get("first_name", "Kullanıcı")

                print(f"🧠 [{gonderen_adi}]: '{ham_metin}' -> Gemini niyet analizi yapıyor...")
                karar = yapay_zeka_niyet_analizi(ham_metin, gonderen_adi)
                niyet = karar.get("niyet", "SOHBET")
                hedef_secim = karar.get("hedef", "AYNI")

                # Kesin kural: Mesajda "bana", "özel", "dm" veya "şahsıma" geçiyorsa doğrudan DM'e zorla
                kucuk_metin = ham_metin.lower()
                if any(k in kucuk_metin for k in ["bana", "özel", "dm", "şahsıma"]):
                    hedef_secim = "DM"

                hedef_id = gonderen_id if hedef_secim == "DM" else (hedef_grup if hedef_secim == "GRUP" else sohbet_id)
                print(f"🎯 Karar -> Niyet: {niyet} | Hedef: {hedef_secim} (Chat ID: {hedef_id})")

                if niyet == "KORELASYON_ANALIZ":
                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": f"🔬 {gonderen_adi}, Seaborn korelasyon ısı haritası ve aylık dağılım analizi hazırlanıyor..."})
                    korelasyon_ve_barplot_analizi_gonder(hedef_chat_id=hedef_id)

                elif niyet == "ML_TAHMIN":
                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": f"🤖 {gonderen_adi}, MinMaxScaler ve Lineer Regresyon modeli eğitiliyor, tahmin hazırlanıyor..."})
                    makine_ogrenmesi_fiyat_tahmini_gonder(hedef_chat_id=hedef_id)

                elif niyet == "ALARM_KUR" and karar.get("hedef_fiyat"):
                    hedef_f = float(karar["hedef_fiyat"])
                    mevcut_gram, _, _ = canli_fiyat_verisi_al()
                    yon = "YUKARI" if hedef_f >= mevcut_gram else "ASAGI"
                    yon_yazi = "yükseldiğinde" if yon == "YUKARI" else "düştüğünde"
                    KULLANICI_ALARMLARI.append({
                        "kisi": gonderen_adi,
                        "chat_id": hedef_id,
                        "hedef_fiyat": hedef_f,
                        "yon": yon
                    })
                    onay_msg = (
                        f"✅ {gonderen_adi}, alarmın kuruldu!\n"
                        f"• Hedef Seviye: {hedef_f:,.2f} TL ({yon_yazi})\n"
                        f"• Şu Anki Fiyat: {mevcut_gram:,.2f} TL\n"
                        f"• Bildirim Yeri: {'Özel Mesaj (DM)' if hedef_secim == 'DM' else 'Grup'}"
                    )
                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": onay_msg})

                elif niyet == "ANLIK_KUR":
                    kur_raporu = anlik_altin_kuru_getir()
                    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_id, "text": kur_raporu})

                elif niyet == "GUNLUK_GRAFIK":
                    if hedef_secim == "DM":
                        bilgi = f"📩 {gonderen_adi}, bugünün saatlik canlı altın grafiği hazırlanıyor; özel mesajınıza gönderilecek."
                    else:
                        bilgi = f"📈 {gonderen_adi}, bugünün saatlik canlı altın grafiği hazırlanıyor..."
                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": bilgi})

                    try:
                        gunluk_canli_grafik_gonder(hedef_chat_id=hedef_id)
                    except Exception as gonderim_hatasi:
                        if hedef_secim == "DM" and "403" in str(gonderim_hatasi):
                            uyari = f"⚠️ {gonderen_adi}, sana özelden mesaj atabilmem için önce bana özel sohbetten /start yazmalısın!"
                            telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": uyari})
                        else:
                            raise gonderim_hatasi

                elif niyet == "TARIH_FIYAT" and karar.get("hedef_tarih"):
                    tarih_str = karar["hedef_tarih"]
                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": f"🔍 {gonderen_adi}, {tarih_str} tarihli piyasa verileri kontrol ediliyor..."})
                    tarih_raporu = gecmis_tarih_fiyat_getir(tarih_str)
                    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_id, "text": tarih_raporu})

                elif niyet == "KAYNAK":
                    kaynak_bilgi = (
                        "📡 VERİ KAYNAĞI VE MİMARİ BİLGİSİ\n\n"
                        "• Canlı & Tarihsel Piyasa Verisi: Yahoo Finance REST API (`GC=F` Ons Altın ve `TRY=X` Dolar/TL)\n"
                        "• Veri Temizleme & İşleme: Python Pandas (`dropna`, `drop_duplicates`, `(Ons * Dolar) / 31.1035`)\n"
                        "• İstatistik & Görselleştirme: Seaborn (`heatmap`, `barplot`) ve Matplotlib\n"
                        "• Makine Öğrenmesi & Yapay Zeka: Scikit-Learn (`MinMaxScaler`, `LinearRegression`) ve Google Gemini LLM API\n\n"
                        "💬 Başka bir isteğiniz var mı?"
                    )
                    telegram_istek_gonder("sendMessage", data={"chat_id": hedef_id, "text": kaynak_bilgi})

                elif niyet == "GRAFIK_ANALIZ":
                    belirli_yil = karar.get("belirli_yil")
                    yil = karar.get("yil_sayisi")
                    donem_etiket = f"{belirli_yil} yılı" if belirli_yil else ("tüm tarihsel" if yil is None else f"son {yil} yıllık")

                    if hedef_secim == "DM":
                        hedef_bilgi = f"📩 {gonderen_adi}, {donem_etiket} analiz ve grafik hazırlanıyor; doğrudan özel mesaj (DM) olarak size gönderilecek."
                    elif hedef_secim == "GRUP":
                        hedef_bilgi = f"📢 {gonderen_adi}, {donem_etiket} analiz ve grafik hazırlanıyor; gruba gönderilecek."
                    else:
                        hedef_bilgi = f"⏳ {donem_etiket.capitalize()} canlı altın verileri çekiliyor ve analiz hazırlanıyor..."

                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": hedef_bilgi})

                    try:
                        boru_hattini_calistir(yil_sayisi=yil, belirli_yil=belirli_yil, hedef_chat_id=hedef_id)
                    except Exception as gonderim_hatasi:
                        if hedef_secim == "DM" and "403" in str(gonderim_hatasi):
                            uyari = f"⚠️ {gonderen_adi}, sana özelden mesaj atabilmem için önce bana özel sohbetten /start yazmalısın!"
                            telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": uyari})
                        else:
                            raise gonderim_hatasi

                else:  # SOHBET
                    sohbet_yaniti = karar.get("sohbet_cevabi") or f"Rica ederim {gonderen_adi}! Başka nasıl yardımcı olabilirim?"
                    telegram_istek_gonder("sendMessage", data={"chat_id": sohbet_id, "text": sohbet_yaniti})

            time.sleep(2)

        except KeyboardInterrupt:
            print("\n🛑 Yummi durduruldu.")
            break
        except Exception as e:
            print(f"⚠️ Dinleme sırasında geçici hata: {e}")
            time.sleep(3)


if __name__ == "__main__":
    telegram_bot_dinle()