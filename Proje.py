import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env dosyasındaki gizli API anahtarlarını yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def finans_verisi_cek(sembol: str, aralik: str = "2y") -> pd.DataFrame:
    """Doğrudan REST API endpoint'ine HTTP GET isteği atarak ham JSON verisini çeker."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sembol}?range={aralik}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    veri = response.json()

    sonuc = veri["chart"]["result"][0]
    zamanlar = sonuc["timestamp"]
    kapanislar = sonuc["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "Tarih": pd.to_datetime(zamanlar, unit="s").date,
        sembol: kapanislar
    })
    return df.dropna()


def veri_isle_ve_gorsellestir() -> tuple[pd.DataFrame, str]:
    print("[1/4] Canlı API'den son 2 yıllık Ons Altın ve Dolar/TL verileri çekiliyor...")
    df_ons = finans_verisi_cek("GC=F", aralik="2y")
    df_kur = finans_verisi_cek("TRY=X", aralik="2y")

    print("[2/4] Pandas ile veriler temizleniyor ve 2025-2026 kıyaslaması yapılıyor...")
    df = pd.merge(df_ons, df_kur, on="Tarih", how="inner")
    df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035
    df["Tarih"] = pd.to_datetime(df["Tarih"])

    df["Yil"] = df["Tarih"].dt.year
    df["Ay"] = df["Tarih"].dt.month
    df = df[df["Yil"].isin([2025, 2026])]

    ozet = df.groupby(["Ay", "Yil"])["Gram_Altin_TL"].mean().unstack()
    ozet.columns = ["2025_Ort_TL", "2026_Ort_TL"]

    ozet["Yillik_Fark_TL"] = ozet["2026_Ort_TL"] - ozet["2025_Ort_TL"]
    ozet["Yillik_Degisim_%"] = (ozet["Yillik_Fark_TL"] / ozet["2025_Ort_TL"]) * 100
    ozet["2026_Aylik_Degisim_%"] = ozet["2026_Ort_TL"].pct_change(fill_method=None) * 100

    ay_isimleri = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }
    ozet.index = ozet.index.map(ay_isimleri)
    ozet = ozet.round(2)

    grafik_yolu = "altin_analiz_raporu.png"
    plt.figure(figsize=(11, 6))
    plt.plot(ozet.index, ozet["2025_Ort_TL"], marker="o", linewidth=2.2, label="2025 Ortalama (TL)")
    plt.plot(ozet.index, ozet["2026_Ort_TL"], marker="s", linewidth=2.2, label="2026 Ortalama (TL)")

    plt.title("2025 vs 2026 Aylık Ortalama Gram Altın Fiyat Kıyaslaması (TL)", fontsize=13, fontweight="bold")
    plt.xlabel("Aylar")
    plt.ylabel("Gram Altın (TL)")
    plt.xticks(rotation=45)
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

    return ozet, grafik_yolu


def llm_ile_analiz_et(ozet_df: pd.DataFrame) -> str:
    print("[3/4] Tablo verisi Gemini LLM API'ye gönderiliyor...")
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=20000)
    )

    tablo_metni = ozet_df.to_string()
    prompt = f"""
    Sen kıdemli bir emtia ve finans analistisin. Aşağıda canlı API'den çekilmiş 2025 ve 2026 yıllarına ait
    aylık ortalama Gram Altın (TL) fiyatları, yıllık farklar ve 2026 içi aylık değişim oranları yer alıyor:

    {tablo_metni}

    Bu verilere dayanarak yatırımcılar için kısa, net ve profesyonel bir rapor hazırla:
    1. Yıl içerisinde altın fiyatlarının hangi dönemlerde/aylarda gevşediğini (düşüş veya yavaşlama gösterdiğini) belirt.
    2. Geçen sene (2025) ile bu sene (2026) arasındaki fiyat ve alım gücü farkını rakamlarla kıyasla.
    3. Dönemsel alım fırsatı kollayan yatırımcılar için 3 maddelik somut çıkarım yap.
    (Not: Telegram mesajı olarak gönderileceği için markdown başlıkları yerine sade madde işaretleri ve kalın vurgular kullan.)
    """

    # Hesabında açık olan en hızlı ve stabil modeller
    adaylar = [
        "gemma-4-31b-it",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.7-flash"
    ]

    for model_adi in adaylar:
        try:
            print(f"   -> Denenen model: {model_adi}")
            response = client.models.generate_content(
                model=model_adi,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"   ⚠️ {model_adi} atlandı ({str(e)[:60]}...)")

    raise RuntimeError("Tüm modeller şu an yoğun.")


def telegrama_gonder(rapor_metni: str, grafik_yolu: str):
    print("[4/4] Analiz raporu ve filigranlı grafik Telegram kanalına gönderiliyor...")
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    with open(grafik_yolu, "rb") as foto:
        res_foto = requests.post(
            f"{base_url}/sendPhoto",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": "📊 2025 vs 2026 Aylık Ortalama Gram Altın Kıyaslama Grafiği"
            },
            files={"photo": foto},
            timeout=15
        )
        res_foto.raise_for_status()

    baslik = "🪙 DÖNEMSEL ALTIN ANALİZ VE KIYASLAMA RAPORU\n\n"
    tam_mesaj = (baslik + rapor_metni)[:4000]

    res_mesaj = requests.post(
        f"{base_url}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": tam_mesaj
        },
        timeout=15
    )
    res_mesaj.raise_for_status()
    print("✅ İşlem tamamlandı! Rapor ve grafik Telegram'a başarıyla iletildi.")


ozet_tablo, grafik_dosyasi = veri_isle_ve_gorsellestir()
print("\n--- PANDAS ÖZET TABLOSU ---")
print(ozet_tablo)

analiz_sonucu = llm_ile_analiz_et(ozet_tablo)
print("\n--- GEMINI ANALİZ ÇIKTISI ---")
print(analiz_sonucu)

telegrama_gonder(analiz_sonucu, grafik_dosyasi)