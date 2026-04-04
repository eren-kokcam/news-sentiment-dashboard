import os
import streamlit as st
import requests
import plotly.graph_objects as go
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv
from datetime import date

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data
def haber_getir(konu,haber_sayisi,tarih,dil_kodu):
    try:
        dil_kodu = "tr" if dil_kodu == "Türkçe" else "en"
        url = f"https://newsapi.org/v2/everything?q={konu}&language={dil_kodu}&pageSize={haber_sayisi}&apiKey={NEWS_API_KEY}&from={tarih}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return None

def ozet_olustur(metin):
    try:
        model= genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            f"Bu haberi 2 cümleyle analiz et, neden önemli olduğunu ve kimi etkilediğini belirt: {metin}"
        )
        return response.text
    except Exception as e:
        st.error(f"Özet oluştururken bir hata oluştu: {e}")
        return "Özet oluşturulamadı."
    
@st.cache_resource
def model_yukle(dil):
    if dil == "Türkçe":
        return pipeline("sentiment-analysis", model="saribasmetehan/bert-base-turkish-sentiment-analysis")
    else:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("Haber Analiz Asistanı")
st.write("Bu uygulama, belirttiğiniz konuda Türkçe haberleri çekerek duygu analizini yapar ve sonuçları görselleştirir. Ayrıca, her haber için özet oluşturma özelliği de sunar.")
konu = st.text_input("Hangi konuyu analız etmek istersiniz?")
haber_sayisi = st.slider("Kaç haber görmek istiyorsunuz?", min_value=5, max_value=20, value=10)
tarih = st.date_input("Baslangic tarihi seçin:", max_value= date.today())
dil = st.selectbox("Haber dili seçin", ["Türkçe", "İngilizce"])

if konu:
    model = model_yukle(dil)
    with st.spinner(f"'{konu}' konusundaki haberler aranıyor..."):
        veriler = haber_getir(konu, haber_sayisi,tarih,dil)
    if veriler is None:
        st.error("Haberler çekilirken bir hata oluştu.")
        st.stop()
    haberler = veriler["articles"]
    if not haberler:
        st.warning("Haber bulunamadı. Lütfen farklı bir konu veya tarih deneyin.")
        st.stop()

    pozitif_sayisi = 0
    negatif_sayisi = 0

    st.info(f"{len(haberler)} haber bulundu.")
    for haber in haberler:
       metin = haber["description"] or haber["title"]
       sonuc = model(metin)
       sonuc_detay = sonuc[0]
       st.subheader(haber["title"])
       if haber["description"] is not None:
           st.write(haber["description"])
       if sonuc_detay["score"] < 0.70:
            st.warning("Belirsiz")
       elif sonuc_detay["label"] == "LABEL_0" or sonuc_detay["label"] == "NEGATIVE":
            st.error("Negatif")
            negatif_sayisi += 1
       elif sonuc_detay["label"] == "LABEL_1" or sonuc_detay["label"] == "POSITIVE":
            st.success("Pozitif")
            pozitif_sayisi += 1
       st.write(f"Duygu Skoru(Doğruluk Yüzdesi): %{sonuc_detay["score"] * 100:.2f}")
       if st.button("Özet Oluştur", key=haber["url"]):
           ozet = ozet_olustur(metin)
           st.write(f"Özet: {ozet}")
       st.markdown(f"[Habere Git]({haber['url']})")
       st.divider()
    st.sidebar.title("Özet:")
    st.sidebar.write(f"Toplam Haber: {len(haberler)}")
    st.sidebar.write(f"Pozitif Haber: {pozitif_sayisi}")
    st.sidebar.write(f"Negatif Haber: {negatif_sayisi}")
    fig = go.Figure(data=[go.Pie(
        labels=["Pozitif", "Negatif"],
        values=[pozitif_sayisi, negatif_sayisi],
        hole=0.3
    )])
    st.sidebar.plotly_chart(fig)
