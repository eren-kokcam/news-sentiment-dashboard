import os
import streamlit as st
import requests
import plotly.graph_objects as go
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data
def haber_getir(konu):
    try:
        url = f"https://newsapi.org/v2/everything?q={konu}&language=tr&pageSize=10&apiKey={NEWS_API_KEY}"
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
def model_yukle():
    return pipeline("sentiment-analysis", model="saribasmetehan/bert-base-turkish-sentiment-analysis")

st.title("Haber Analiz Asistanı")
st.write("Bu uygulama, belirttiğiniz konuda Türkçe haberleri çekerek duygu analizini yapar ve sonuçları görselleştirir. Ayrıca, her haber için özet oluşturma özelliği de sunar.")
konu = st.text_input("Hangi konuyu analız etmek istersiniz?")

if konu:
    model = model_yukle()
    with st.spinner(f"'{konu}' konusundaki haberler aranıyor..."):
        veriler = haber_getir(konu)
    if veriler is None:
        st.error("Haberler çekilirken bir hata oluştu.")
        st.stop()
    haberler = veriler["articles"]
    if not haberler:
        st.write("Bu konuda haber bulunamadı.")
        st.stop()
    pozitif_sayisi = 0
    negatif_sayisi = 0

    for haber in haberler:
       metin = haber["description"] or haber["title"]
       sonuc = model(metin)
       sonuc_detay = sonuc[0]
       st.subheader(haber["title"])
       if haber["description"] is not None:
           st.write(haber["description"])
       if sonuc_detay["label"] == "LABEL_0":
           st.error("Negatif")
           negatif_sayisi += 1
       elif sonuc_detay["label"] == "LABEL_1":
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
