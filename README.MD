# Haber Duygu Analizi Dashboard

T�rk�e haberlerin duygu analizini yapan Streamlit tabanl1 bir web uygulamas1.

## �zellikler

- NewsAPI �zerinden ger�ek zamanl1 haber arama
- BERT tabanl1 T�rk�e duygu analizi modeli
- Pozitif/Negatif duygu s1n1fland1rmas1
- G�ven skoru y�zdesi
- Kullan1c1 dostu aray�z

## Gereksinimler

```bash
pip install streamlit requests transformers python-dotenv torch
```

## Kurulum

1. Repoyu klonlay1n:
```bash
git clone <repo-url>
cd news-sentiment-dashboard
```

2. Sanal ortam olu_turun ve aktif edin:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate  # Windows
```

3. Gerekli paketleri y�kleyin:
```bash
pip install streamlit requests transformers python-dotenv torch
```

4. `.env` dosyas1 olu_turun ve NewsAPI anahtar1n1z1 ekleyin:
```
NEWS_API_KEY=your_api_key_here
```

NewsAPI anahtar1 almak i�in: https://newsapi.org/

## Kullan1m

Uygulamay1 ba_lat1n:
```bash
streamlit run app.py
```

Taray1c1n1zda otomatik olarak a�1lacak olan aray�zde:
1. Analiz etmek istediğiniz konuyu girin
2. Uygulama ilgili T�rk�e haberleri �ekecek
3. Her haber i�in duygu analizi sonu�lar1 g�r�nt�lenecek

## Kullan1lan Teknolojiler

- **Streamlit**: Web arayüzü
- **NewsAPI**: Haber kaynağ1
- **Transformers**: Duygu analizi modeli
- **BERT Model**: `saribasmetehan/bert-base-turkish-sentiment-analysis`

## Model Bilgisi

Uygulama, T�rk�e metinler i�in �zel olarak eğitilmi_ BERT tabanl1 bir sentiment analysis modeli kullan1r:
- Model: `saribasmetehan/bert-base-turkish-sentiment-analysis`
- �1kt1: LABEL_0 (Negatif) veya LABEL_1 (Pozitif)
- G�ven skoru: 0-1 aras1 (y�zde olarak g�sterilir)

## Lisans

MIT
