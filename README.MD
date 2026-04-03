# Haber Analiz Asistanı

Türkçe haberlerin duygu analizini yapan ve özetleyen Streamlit tabanlı web uygulaması.

## Özellikler

- NewsAPI üzerinden gerçek zamanlı haber arama
- BERT tabanlı Türkçe duygu analizi modeli
- Pozitif/Negatif/Belirsiz duygu sınıflandırması
- Güven skoru yüzdesi gösterimi
- **Gemini AI ile haber özeti oluşturma**
- **Tarih bazlı haber filtreleme**
- **Haber sayısı seçimi (5-20 arası)**
- **İnteraktif sidebar özet grafiği (Plotly)**
- **Pozitif/Negatif haber dağılımı pasta grafiği**
- Kullanıcı dostu arayüz

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

4. `.env` dosyası oluşturun ve API anahtarlarınızı ekleyin:
```
NEWS_API_KEY=your_newsapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

API anahtarları:
- NewsAPI: https://newsapi.org/
- Google Gemini: https://ai.google.dev/

## Kullanım

Uygulamayı başlatın:
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacak olan arayüzde:
1. Analiz etmek istediğiniz konuyu girin
2. Kaç haber görmek istediğinizi seçin (5-20 arası)
3. Başlangıç tarihini belirleyin
4. Uygulama ilgili Türkçe haberleri çekecek
5. Her haber için duygu analizi sonuçları görüntülenecek
6. İstediğiniz haber için "Özet Oluştur" butonuna tıklayarak AI destekli özet alabilirsiniz
7. Sidebar'da genel istatistikleri ve pasta grafiğini görebilirsiniz

## Kullanılan Teknolojiler

- **Streamlit**: Web arayüzü ve veri görselleştirme
- **NewsAPI**: Haber kaynağı API
- **Transformers (Hugging Face)**: Duygu analizi modeli
- **BERT Model**: `saribasmetehan/bert-base-turkish-sentiment-analysis`
- **Google Gemini AI**: Haber özetleme (gemini-2.5-flash)
- **Plotly**: İnteraktif grafik görselleştirme

## Model Bilgisi

### Duygu Analizi Modeli
Uygulama, Türkçe metinler için özel olarak eğitilmiş BERT tabanlı bir sentiment analysis modeli kullanır:
- **Model**: `saribasmetehan/bert-base-turkish-sentiment-analysis`
- **Çıktı**:
  - LABEL_0: Negatif
  - LABEL_1: Pozitif
  - Belirsiz: Güven skoru %70'in altında olan haberler
- **Güven skoru**: 0-1 arası (yüzde olarak gösterilir)

### AI Özet Motoru
- **Model**: Google Gemini 2.5 Flash
- **Görev**: Her haber için 2 cümlelik analiz ve özet oluşturma
- **Özellik**: Haberin neden önemli olduğunu ve kimi etkilediğini açıklar

## Lisans

MIT
