# Haber Duygu Analizi Dashboard

Türkçe haberlerin duygu analizini yapan Streamlit tabanlı bir web uygulaması.

## Özellikler

- NewsAPI üzerinden gerçek zamanlı haber arama
- BERT tabanlı Türkçe duygu analizi modeli
- Pozitif/Negatif duygu sınıflandırması
- Güven skoru yüzdesi
- Kullanıcı dostu arayüz

## Gereksinimler

```bash
pip install streamlit requests transformers python-dotenv torch
```

## Kurulum

1. Repoyu klonlayın:
```bash
git clone <repo-url>
cd news-sentiment-dashboard
```

2. Sanal ortam oluşturun ve aktif edin:
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate  # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. `.env` dosyası oluşturun ve NewsAPI anahtarınızı ekleyin:
```
NEWS_API_KEY=your_api_key_here
```

NewsAPI anahtarı almak için: https://newsapi.org/

## Kullanım

Uygulamayı başlatın:
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacak olan arayüzde:
1. Analiz etmek istediğiniz konuyu girin
2. Uygulama ilgili Türkçe haberleri çekecek
3. Her haber için duygu analizi sonuçları görüntülenecek

## Kullanılan Teknolojiler

- **Streamlit**: Web arayüzü
- **NewsAPI**: Haber kaynağı
- **Transformers**: Duygu analizi modeli
- **BERT Model**: `saribasmetehan/bert-base-turkish-sentiment-analysis`

## Model Bilgisi

Uygulama, Türkçe metinler için özel olarak eğitilmiş BERT tabanlı bir sentiment analysis modeli kullanır:
- Model: `saribasmetehan/bert-base-turkish-sentiment-analysis`
- Çıktı: LABEL_0 (Negatif) veya LABEL_1 (Pozitif)
- Güven skoru: 0-1 arası (yüzde olarak gösterilir)

## Lisans

MIT
