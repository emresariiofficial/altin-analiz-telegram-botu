import os
import json
import time
from typing import Optional
from datetime import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env dosyasındaki gizli API anahtarlarını yükle
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GRUP_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
    """Gelen her mesajı Gemini'ye inceletir; niyeti, belirli yılı, yıl sayısını veya sohbet cevabını JSON döndürür."""
    bugun_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    Bugünün tarihi: {bugun_str}.
    Sen Telegram'da çalışan 'Yummi' adında samimi, zeki ve profesyonel bir Altın Analiz Asistanısın.
    Kullanıcı ({gonderen_adi}) sana şu mesajı yazdı: "{kullanici_mesaji}"

    Bu mesajı incele ve SADECE aşağıdaki şemaya uygun geçerli bir JSON objesi döndür:
    {{
        "niyet": "ANLIK_KUR" | "TARIH_FIYAT" | "GRAFIK_ANALIZ" | "KAYNAK" | "SOHBET",
        "hedef_tarih": "<Eğer niyet TARIH_FIYAT ise YYYY-MM-DD formatında kesin tarih (örn: '2025-06-30'), değilse null>",
        "belirli_yil": <Eğer kullanıcı '2023 yılının grafiği', '2020 altın grafiği' gibi TEK BİR SPESİFİK TAKVİM YILI istediyse o yıl (örn: 2023), değilse null>,
        "yil_sayisi": <Eğer belirli_yil null ise ve 'son 5 yıl' gibi bir aralık istendiyse tam sayı (örn: 5), 'max/tüm yıllar' istendiyse null, belirtilmediyse 2>,
        "hedef": "DM" | "GRUP" | "AYNI",
        "sohbet_cevabi": "<Eğer niyet SOHBET ise {gonderen_adi} kullanıcısına Yummi karakteriyle vereceğin samimi, kısa ve yardımcı Türkçe cevap; değilse boş string>"
    }}

    Kurallar:
    - '30.06.2025 de altın ne kadardı', 'geçen sene 30 haziranda neydi' gibi GÜN/AY/YIL içeren tek bir günün fiyatında niyet = "TARIH_FIYAT" ve hedef_tarih = "YYYY-MM-DD".
    - '2023 yılının altın grafiği', '2021 grafiğini at' gibi belirli bir yılın grafiği isteniyorsa niyet = "GRAFIK_ANALIZ" ve belirli_yil = 2023 (istenen yıl) olsun.
    - 'son 5 yılın grafiği', 'tüm yılları kıyasla' gibi çok yıllı isteklerde niyet = "GRAFIK_ANALIZ", belirli_yil = null ve yil_sayisi = 5 (veya max için null).
    - 'bugün altın ne kadar', 'güncel kur' sorularında niyet = "ANLIK_KUR".
    - 'kaynağın ne', 'veriler nereden' sorularında niyet = "KAYNAK".
    - 'teşekkürler', 'sağ ol', 'merhaba', 'nasılsın' veya genel sohbetlerde niyet = "SOHBET".
    - Mesajda 'bana at', 'bana gönder', 'özelden', 'dm' geçiyorsa hedef = "DM"; 'gruba' geçiyorsa "GRUP"; aksi halde "AYNI".
    """
    try:
        ham_yanit = gemini_metin_uret(prompt, json_modu=True)
        temiz_json = ham_yanit.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(temiz_json)
    except Exception as e:
        print(f"⚠️ Niyet analizi yedek moda düştü: {e}")
        return {
            "niyet": "SOHBET",
            "hedef_tarih": None,
            "belirli_yil": None,
            "yil_sayisi": 2,
            "hedef": "AYNI",
            "sohbet_cevabi": f"Rica ederim {gonderen_adi}! Ben Yummi, her zaman buradayım. Güncel altın kurunu, geçmiş bir tarihteki fiyatı veya istediğin yıl aralığının grafiğini sorabilirsin."
        }


def finans_verisi_cek(sembol: str, aralik: str = "2y") -> pd.DataFrame:
    """Doğrudan REST API endpoint'ine HTTP GET isteği atarak istenen aralıkta ham JSON verisini çeker."""
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


def gecmis_tarih_fiyat_getir(hedef_tarih_str: str) -> str:
    """Belirli bir geçmiş tarihteki (hafta sonuysa en yakın önceki işlem günündeki) Gram Altın fiyatını getirir."""
    try:
        hedef_dt = datetime.strptime(hedef_tarih_str, "%Y-%m-%d").date()
        if hedef_dt > datetime.now().date():
            return f"⚠️ {hedef_dt.strftime('%d.%m.%Y')} henüz gelmemiş bir tarih! Geçmiş bir tarih veya güncel kur sorabilirsin."

        fark_yil = datetime.now().year - hedef_dt.year
        aralik = "1y" if fark_yil <= 1 else ("5y" if fark_yil <= 5 else "max")

        df_ons = finans_verisi_cek("GC=F", aralik=aralik)
        df_kur = finans_verisi_cek("TRY=X", aralik=aralik)
        df = pd.merge(df_ons, df_kur, on="Tarih", how="inner")
        df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035

        uygun_kayitlar = df[df["Tarih"] <= hedef_dt]
        if uygun_kayitlar.empty:
            return f"⚠️ {hedef_dt.strftime('%d.%m.%Y')} tarihi için API arşivinde veri bulunamadı."

        secilen = uygun_kayitlar.iloc[-1]
        gercek_tarih = secilen["Tarih"].strftime("%d.%m.%Y")
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
    headers = {"User-Agent": "Mozilla/5.0"}
    res_ons = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=1d&interval=1m", headers=headers, timeout=10).json()
    res_kur = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/TRY=X?range=1d&interval=1m", headers=headers, timeout=10).json()

    ons_fiyat = res_ons["chart"]["result"][0]["meta"]["regularMarketPrice"]
    kur_fiyat = res_kur["chart"]["result"][0]["meta"]["regularMarketPrice"]
    gram_altin = (ons_fiyat * kur_fiyat) / 31.1035
    zaman_damgasi = datetime.now().strftime("%d.%m.%Y %H:%M")

    return (
        f"💰 GÜNCEL CANLI ALTIN VE DÖVİZ KURLARI ({zaman_damgasi})\n\n"
        f"• Gram Altın (TL): {gram_altin:,.2f} ₺\n"
        f"• Ons Altın (USD): {ons_fiyat:,.2f} $\n"
        f"• Dolar / TL: {kur_fiyat:,.4f} ₺\n\n"
        f"ℹ️ Veriler canlı finans API'si üzerinden anlık hesaplanmıştır.\n\n"
        f"💬 Başka bir isteğiniz var mı? (Örn: 'son 5 yılın grafiğini gruba at' veya 'bana son 20 yılın grafiğini at')"
    )


def veri_isle_ve_gorsellestir(yil_sayisi: Optional[int] = 2, belirli_yil: Optional[int] = None) -> tuple:
    guncel_yil = datetime.now().year

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
    df = pd.merge(df_ons, df_kur, on="Tarih", how="inner")
    df["Gram_Altin_TL"] = (df["GC=F"] * df["TRY=X"]) / 31.1035
    df["Tarih"] = pd.to_datetime(df["Tarih"])

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


def telegrama_gonder(rapor_metni: str, grafik_yolu: str, hedef_chat_id: str, mevcut_yillar: list):
    donem_baslik = str(mevcut_yillar[0]) if len(mevcut_yillar) == 1 else f"{mevcut_yillar[0]}-{mevcut_yillar[-1]}"
    print(f"[4/4] Analiz raporu ve grafik hedef sohbete ({hedef_chat_id}) gönderiliyor...")
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    with open(grafik_yolu, "rb") as foto:
        res_foto = requests.post(
            f"{base_url}/sendPhoto",
            data={
                "chat_id": hedef_chat_id,
                "caption": f"📊 {donem_baslik} Dönemi Aylık Ortalama Gram Altın Grafiği"
            },
            files={"photo": foto},
            timeout=15
        )
        res_foto.raise_for_status()

    devam_notu = "\n\n💬 Başka bir analiz veya grafik ister misiniz? (Örn: 'son 10 yılın grafiğini gruba at' veya '2024 yılının grafiğini at')"
    baslik = f"🪙 {donem_baslik} ({len(mevcut_yillar)} YILLIK) ALTIN ANALİZ RAPORU\n\n"
    tam_mesaj = (baslik + rapor_metni)[:3850] + devam_notu

    res_mesaj = requests.post(
        f"{base_url}/sendMessage",
        data={
            "chat_id": hedef_chat_id,
            "text": tam_mesaj
        },
        timeout=15
    )
    res_mesaj.raise_for_status()
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
    """Açılışta ve gruba yeni biri katıldığında Yummi olarak kendini tanıtır, tüm mesajları Gemini AI Router ile yorumlar."""
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    son_update_id = None

    ilk_yanit = requests.get(f"{base_url}/getUpdates", timeout=15).json()
    if ilk_yanit.get("result"):
        son_update_id = ilk_yanit["result"][-1]["update_id"] + 1

    karsilama_mesaji = (
        "Selam Hoşgeldiniz! Ben Altın Analiz Asistanınız Yummi, size nasıl yardımcı olabilirim?\n\n"
        "💡 Neler Yapabilirim? (Doğrudan yazmanız yeterli):\n"
        "• güncel altın kuru ne (Anlık Gram, Ons ve Dolar kurunu söyler)\n"
        "• 30.06.2025 tarihinde altın ne kadardı (İstediğiniz geçmiş tarihin fiyatını söyler)\n"
        "• 2023 yılının altın grafiğini at (Belirli bir yılın grafiğini çizer)\n"
        "• son 5 yılın grafiğini gruba at (Çok yıllı kıyaslama yapar)\n"
        "• bana son 20 yılın grafiğini at (Özel mesajınıza atar)\n"
        "• /analiz max (En eski tarihten bugüne tüm yılları kıyaslar)"
    )

    hedef_grup = str(GRUP_CHAT_ID).strip() if GRUP_CHAT_ID else ""
    print(f"📡 Hedef Grup ID (.env): '{hedef_grup}'")

    if hedef_grup:
        res_acilis = requests.post(
            f"{base_url}/sendMessage",
            data={"chat_id": hedef_grup, "text": karsilama_mesaji},
            timeout=10
        )
        if res_acilis.status_code == 200:
            print("✅ Karşılama mesajı gruba başarıyla gönderildi!")
        else:
            print(f"❌ Karşılama mesajı gönderilemedi! Hata kodu: {res_acilis.status_code} | Detay: {res_acilis.text}")

    print("\n🐱 Yummi Aktif! Yapay Zeka Niyet Yönlendiricisi devrede, mesajlarınız bekleniyor... (Durdurmak için Ctrl+C)\n")

    while True:
        try:
            params = {"timeout": 10}
            if son_update_id:
                params["offset"] = son_update_id

            yanit = requests.get(f"{base_url}/getUpdates", params=params, timeout=20).json()
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
                            requests.post(f"{base_url}/sendMessage", data={"chat_id": sohbet_id, "text": hosgeldin}, timeout=10)
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

                hedef_id = gonderen_id if hedef_secim == "DM" else (hedef_grup if hedef_secim == "GRUP" else sohbet_id)

                if niyet == "ANLIK_KUR":
                    kur_raporu = anlik_altin_kuru_getir()
                    requests.post(f"{base_url}/sendMessage", data={"chat_id": hedef_id, "text": kur_raporu}, timeout=10)

                elif niyet == "TARIH_FIYAT" and karar.get("hedef_tarih"):
                    tarih_str = karar["hedef_tarih"]
                    requests.post(f"{base_url}/sendMessage", data={"chat_id": sohbet_id, "text": f"🔍 {gonderen_adi}, {tarih_str} tarihli piyasa verileri kontrol ediliyor..."}, timeout=10)
                    tarih_raporu = gecmis_tarih_fiyat_getir(tarih_str)
                    requests.post(f"{base_url}/sendMessage", data={"chat_id": hedef_id, "text": tarih_raporu}, timeout=10)

                elif niyet == "KAYNAK":
                    kaynak_bilgi = (
                        "📡 VERİ KAYNAĞI VE MİMARİ BİLGİSİ\n\n"
                        "• Canlı & Tarihsel Piyasa Verisi: Yahoo Finance REST API (`GC=F` Ons Altın ve `TRY=X` Dolar/TL)\n"
                        "• Veri İşleme & Hesaplama: Python Pandas (`(Ons * Dolar) / 31.1035`)\n"
                        "• Görselleştirme: Matplotlib\n"
                        "• Yapay Zeka Analizi & Niyet Yönlendirme: Google Gemini LLM API\n\n"
                        "💬 Başka bir isteğiniz var mı?"
                    )
                    requests.post(f"{base_url}/sendMessage", data={"chat_id": hedef_id, "text": kaynak_bilgi}, timeout=10)

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

                    requests.post(f"{base_url}/sendMessage", data={"chat_id": sohbet_id, "text": hedef_bilgi}, timeout=10)

                    try:
                        boru_hattini_calistir(yil_sayisi=yil, belirli_yil=belirli_yil, hedef_chat_id=hedef_id)
                    except Exception as gonderim_hatasi:
                        if hedef_secim == "DM" and "403" in str(gonderim_hatasi):
                            uyari = f"⚠️ {gonderen_adi}, sana özelden mesaj atabilmem için önce bana özel sohbetten /start yazmalısın!"
                            requests.post(f"{base_url}/sendMessage", data={"chat_id": sohbet_id, "text": uyari}, timeout=10)
                        else:
                            raise gonderim_hatasi

                else:  # SOHBET
                    sohbet_yaniti = karar.get("sohbet_cevabi") or f"Rica ederim {gonderen_adi}! Başka nasıl yardımcı olabilirim?"
                    requests.post(f"{base_url}/sendMessage", data={"chat_id": sohbet_id, "text": sohbet_yaniti}, timeout=10)

        except KeyboardInterrupt:
            print("\n🛑 Yummi durduruldu.")
            break
        except Exception as e:
            print(f"⚠️ Dinleme sırasında geçici hata: {e}")
            time.sleep(3)


if __name__ == "__main__":
    telegram_bot_dinle()