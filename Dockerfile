# Python 3.11.9 temel imajı
FROM python:3.11.9-slim-bookworm

# Çalışma dizini oluştur
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# NLTK verilerini indir
RUN python -m nltk.downloader punkt stopwords

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Ollama modelini önceden yükle (opsiyonel)
RUN ollama pull llama3

# Uygulamayı çalıştır
CMD ["python", "main.py"]