# NTT DATA Sürdürülebilirlik Raporları RAG Sistemi

Bu proje, NTT DATA'nın sürdürülebilirlik raporları üzerinde RAG (Retrieval-Augmented Generation) tabanlı sorgulama yapabilen bir sistem geliştirmektedir. Sistem, PDF raporlarını otomatik olarak indirip işleyerek vektör veritabanına kaydeder ve kullanıcıların doğal dil ile sorular sorabilmesini sağlar.

## 📋 İçindekiler

- [Özellikler](#Özellikler)
- [Teknoloji Stack](#teknoloji-stack)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [API Dökümantasyonu](#api-dökümantasyonu)
- [Konfigürasyon](#konfigürasyon)
- [Geliştirme](#geliştirme)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🚀 Özellikler

### Core RAG Özellikleri
- **Otomatik PDF İndirme**: NTT DATA sürdürülebilirlik raporlarını otomatik indirme
- **Akıllı Metin İşleme**: PDF'lerden metin çıkarma, temizleme ve chunking
- **Gelişmiş Embedding**: Sentence-Transformers ile yüksek kaliteli embeddings
- **Vektör Veritabanı**: ChromaDB ile hızlı ve ölçeklenebilir vektör arama
- **Hibrit Retrieval**: Vektör benzerliği + BM25 sparse retrieval kombinasyonu

### Gelişmiş RAG Teknikleri
- **HyDE (Hypothetical Document Embeddings)**: Daha iyi retrieval için hipotez belge oluşturma
- **Multi-Query Retrieval**: Tek soru için çoklu sorgu variant'ları oluşturma
- **Ensemble Retrieval**: Farklı retrieval yöntemlerini birleştirme
- **Contextual Compression**: LLM ile irrelevant içeriği filtreleme
- **Dynamic Reranking**: Sonuçları bağlama göre yeniden sıralama

### Teknik Özellikler
- **FastAPI**: RESTful API arayüzü
- **Asenkron İşleme**: Yüksek performanslı concurrent operations
- **Docker Support**: Containerized deployment
- **GCP Integration**: Google Cloud Storage ile backup ve scaling
- **Monitoring**: Detaylı logging ve metrics
- **Health Checks**: Sistem durumu kontrolü

## 🛠 Teknoloji Stack

### Backend & API
- **FastAPI**: Modern, hızlı web framework
- **Python 3.11**: Temel programlama dili
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation ve serialization

### NLP & ML
- **Sentence-Transformers**: Embedding oluşturma
- **NLTK**: Doğal dil işleme
- **scikit-learn**: ML utilities
- **NumPy**: Numerical computing

### RAG & LLM
- **LangChain**: RAG pipeline framework
- **Ollama**: Yerel LLM inference
- **Llama 3**: Language model
- **ChromaDB**: Vektör veritabanı

### PDF İşleme
- **PyMuPDF**: PDF text extraction
- **Regular Expressions**: Text cleaning

### Cloud & Storage
- **Google Cloud Storage**: Backup ve storage
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

### Monitoring & Testing
- **Loguru**: Advanced logging
- **pytest**: Unit testing
- **HTTPX**: HTTP client for testing

## 📦 Kurulum

### Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- Ollama (yerel LLM için)
- Google Cloud hesabı (opsiyonel, backup için)

### 1. Repository'yi Klonla

```bash
git clone https://github.com/furkan-cyber/ntt-sustainability-rag.git
cd ntt-sustainability-rag
```

### 2. Environment Ayarları

`.env` dosyasını düzenle:

```env
# GCP Konfigürasyonu
GCP_BUCKET_NAME=ntt-sustainability-reports
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# LLM Ayarları
LLM_MODEL=llama3
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000

# Vektör Veritabanı Ayarları
CHROMA_DB_DIR=/app/data/vector_db
COLLECTION_NAME=ntt_sustainability_reports

# FastAPI Ayarları
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_RELOAD=False

# RAG Ayarları
TOP_K_RETRIEVAL=5
CONFIDENCE_THRESHOLD=0.7
ENABLE_HYDE=True
ENABLE_MULTI_QUERY=True
ENABLE_RERANK=True
```

### 3. Docker ile Kurulum (Önerilen)

```bash
# Servisleri başlat
docker-compose up -d

# Logları kontrol et
docker-compose logs -f app
```

### 4. Manuel Kurulum

```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# NLTK verilerini indir
python -m nltk.downloader punkt stopwords

# Ollama'yı başlat (ayrı terminal)
ollama serve

# Llama3 modelini yükle
ollama pull llama3

# Uygulamayı çalıştır
python main.py
```

## 🎯 Kullanım

### API Servisi Başlatma

Sistem başlatıldığında otomatik olarak:
1. NTT DATA sürdürülebilirlik raporlarını indirir
2. PDF'leri işleyerek vektör veritabanına kaydeder
3. FastAPI server'ını başlatır
4. Sağlık kontrollerini aktif eder

### Temel Sorgu Örneği

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "NTT DATA'nın 2023 yılındaki karbon emisyon hedefleri nelerdir?"}'
```

### Python ile Kullanım

```python
import requests

# Soru sor
response = requests.post("http://localhost:8000/ask", json={
    "question": "NTT DATA'nın sürdürülebilirlik stratejisi hakkında bilgi ver"
})

result = response.json()
print(f"Cevap: {result['answer']}")
print(f"Kaynaklar: {result['sources']}")
```

## 📚 API Dökümantasyonu

### Endpoints

#### POST /ask
Sürdürülebilirlik raporları hakkında soru sor.

**Request Body:**
```json
{
  "question": "string"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "source": "string",
      "year": "integer",
      "score": "float",
      "method": "string"
    }
  ]
}
```

#### GET /health
Sistem sağlık durumunu kontrol et.

**Response:**
```json
{
  "status": "healthy",
  "vector_db_count": 1250,
  "last_updated": "2024-01-15T10:30:00"
}
```

### Örnek Sorular

- "NTT DATA'nın 2023 yılındaki CO2 emisyon hedefleri nelerdir?"
- "Şirketin yeşil enerji kullanımı hakkında bilgi ver"
- "Çalışan memnuniyeti ile ilgili metrikler neler?"
- "Sürdürülebilirlik alanındaki yatırımlar hangi alanlara odaklanıyor?"
- "Döngüsel ekonomi yaklaşımları neler?"

## ⚙️ Konfigürasyon

### Temel Ayarlar

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `CHUNK_SIZE` | Metin chunk boyutu | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `EMBEDDING_MODEL` | Embedding model | all-mpnet-base-v2 |
| `TOP_K_RETRIEVAL` | Retrieval sonuç sayısı | 5 |
| `CONFIDENCE_THRESHOLD` | Minimum güven eşiği | 0.7 |

### RAG Teknikleri

| Özellik | Açıklama | Varsayılan |
|---------|----------|------------|
| `ENABLE_HYDE` | HyDE tekniği | True |
| `ENABLE_MULTI_QUERY` | Multi-query retrieval | True |
| `ENABLE_RERANK` | Yeniden sıralama | True |

### LLM Ayarları

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `LLM_MODEL` | Ollama model adı | llama3 |
| `LLM_TEMPERATURE` | Creativity level | 0.3 |
| `LLM_MAX_TOKENS` | Max token sayısı | 2000 |

## 🔧 Geliştirme

### Proje Yapısı

```
ntt-sustainability-rag/
├── main.py                 # Ana uygulama
├── requirements.txt        # Python bağımlılıkları
├── .env                   # Environment variables
├── Dockerfile             # Docker image
├── docker-compose.yml     # Multi-container setup
├── README.md             # Bu dosya
├── data/                 # Veri dizini
│   ├── pdfs/            # İndirilen PDF'ler
│   ├── processed/       # İşlenmiş veriler
│   └── vector_db/       # Vektör veritabanı
└── tests/               # Test dosyaları
    ├── __init__.py
    ├── test_api.py
    ├── test_rag.py
    └── test_utils.py
```

### Sınıf Yapısı

```python
# Ana modüller
Config                    # Konfigürasyon yönetimi
DocumentProcessor        # PDF işleme
VectorDatabase          # Vektör veritabanı
LLMService             # LLM etkileşimi
RAGPipeline            # RAG pipeline
APIService             # FastAPI servisi
DataIngestion          # Veri yükleme
GCPIntegration         # GCP entegrasyon
Monitoring             # Sistem izleme
```

### Test Çalıştırma

```bash
# Tüm testler
pytest

# Belirli test dosyası
pytest tests/test_api.py

# Coverage ile
pytest --cov=main

# Verbose output
pytest -v
```

### Yeni Özellik Ekleme

1. **Yeni Retrieval Yöntemi**:
```python
class CustomRetriever:
    def retrieve(self, query: str) -> List[Dict]:
        # Yeni retrieval logic
        pass
```

2. **RAGPipeline'a Entegrasyon**:
```python
# RAGPipeline.retrieve_documents() metoduna ekle
custom_results = self.custom_retriever.retrieve(question)
retrieval_methods.append(("custom", custom_results))
```

## 🚀 Deployment

### Docker Production Deployment

```bash
# Production build
docker build -t ntt-sustainability-rag:latest .

# Production run
docker run -d \
  --name ntt-rag \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e GCP_BUCKET_NAME=your-bucket \
  ntt-sustainability-rag:latest
```

### Google Cloud Platform

1. **Container Registry'e Push**:
```bash
# Image'i tag'le
docker tag ntt-sustainability-rag:latest gcr.io/YOUR-PROJECT/ntt-sustainability-rag

# Push et
docker push gcr.io/YOUR-PROJECT/ntt-sustainability-rag
```

2. **Cloud Run Deploy**:
```bash
gcloud run deploy ntt-sustainability-rag \
  --image gcr.io/YOUR-PROJECT/ntt-sustainability-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s
```

### Environment Variables (Production)

```env
# Production ayarları
FASTAPI_RELOAD=False
LLM_TEMPERATURE=0.2
CONFIDENCE_THRESHOLD=0.8
ENABLE_RERANK=True

# Scaling ayarları
TOP_K_RETRIEVAL=10
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## 🔍 Troubleshooting

### Yaygın Sorunlar

#### 1. Ollama Connection Error
```bash
# Ollama servisini kontrol et
ollama list

# Modeli yeniden yükle
ollama pull llama3

# Servisi yeniden başlat
ollama serve
```

#### 2. ChromaDB Lock Error
```bash
# Vektör veritabanını sıfırla
rm -rf data/vector_db/*

# Uygulamayı yeniden başlat
python main.py
```

#### 3. Memory Issues
```python
# Config ayarlarını optimize et
CHUNK_SIZE = 500
TOP_K_RETRIEVAL = 3
ENABLE_HYDE = False
```

#### 4. PDF Download Failures
```python
# Network timeout'u artır
response = requests.get(url, timeout=30)

# Retry logic ekle
for attempt in range(3):
    try:
        response = requests.get(url, timeout=10)
        break
    except requests.exceptions.Timeout:
        time.sleep(2)
```

### Debugging

```bash
# Verbose logging
export PYTHONPATH=$PYTHONPATH:.
python -m logging.basicConfig level=DEBUG main.py

# API endpoint test
curl -X GET "http://localhost:8000/health"

# Container logs
docker-compose logs -f app
```

### Performance Tuning

1. **Embedding Optimizasyonu**:
```python
# GPU kullanımı (varsa)
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

# Batch processing
embeddings = model.encode(texts, batch_size=32)
```

2. **Vektör Veritabanı Optimizasyonu**:
```python
# Index parametrelerini ayarla
collection = client.create_collection(
    name="optimized_collection",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 16
    }
)
```

## 📊 Monitoring ve Metrics

### Sistem Metrikleri

- **Response Time**: Ortalama yanıt süresi
- **Retrieval Accuracy**: Alınan belgelerin relevance skoru
- **Token Usage**: LLM token tüketimi
- **Vector DB Size**: Vektör veritabanı boyutu
- **Query Patterns**: Sık sorulan sorular

### Monitoring Endpoints

```bash
# Sağlık durumu
curl http://localhost:8000/health

# Metrics (future feature)
curl http://localhost:8000/metrics
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

## 📄 License

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 🆘 Support

- **Issues**: GitHub Issues kullanın
- **Documentation**: README.md
- **Email**: furkanavcioglu11@gmail.com

## 🔄 Changelog

### v1.0.0
- İlk release
- Temel RAG functionality
- FastAPI integration
- Docker support
- GCP integration

### Future Roadmap
- [ ] Web UI interface
- [ ] Multi-language support
- [ ] Advanced caching
- [ ] Real-time updates
- [ ] Enhanced monitoring
- [ ] A/B testing framework
