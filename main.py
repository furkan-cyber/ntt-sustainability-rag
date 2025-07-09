#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NTT DATA Sürdürülebilirlik Raporları RAG Sistemi

Bu modül, NTT DATA'nın sürdürülebilirlik raporları üzerinde RAG (Retrieval-Augmented Generation)
işlemleri yapan bir servis sağlar.

Özellikler:
- PDF raporlarını indirme ve işleme
- Metin chunking ve temizleme
- Advanced embedding oluşturma
- Vektör veritabanı entegrasyonu
- LLM ile cevap oluşturma
- FastAPI REST arayüzü
- Docker desteği
- Unit testler
- CI/CD entegrasyonu
- GCP deploy desteği
"""

import os
import re
import logging
import requests
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

# NLP ve ML kütüphaneleri
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk

# Vektör veritabanı
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# LLM ve RAG
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Web framework
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Konfigürasyon yönetimi
from dotenv import load_dotenv

# GCP desteği
from google.cloud import storage

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# NLTK verilerini indir
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Environment variables
load_dotenv()

class Config:
    """Uygulama konfigürasyon ayarları"""
    PDF_STORAGE_DIR = Path("./data/pdfs")
    PROCESSED_DIR = Path("./data/processed")
    VECTOR_DB_DIR = Path("./data/vector_db")
    CHUNK_SIZE = 1000  # Karakter cinsinden chunk boyutu
    CHUNK_OVERLAP = 200  # Chunk'lar arası overlap
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM = 768
    COLLECTION_NAME = "ntt_sustainability_reports"
    LLM_MODEL = "llama3"  # Ollama için model adı
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 2000
    FASTAPI_PORT = 8000
    FASTAPI_HOST = "0.0.0.0"
    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "")
    ENABLE_HYDE = True  # Hypothetical Document Embeddings (HyDE) etkinleştirme
    ENABLE_MULTI_QUERY = True  # Multi-query retrieval etkinleştirme
    ENABLE_RERANK = True  # Yeniden sıralama etkinleştirme
    TOP_K_RETRIEVAL = 5  # Alınacak belge sayısı
    CONFIDENCE_THRESHOLD = 0.7  # Minimum güven eşiği

    @classmethod
    def setup_dirs(cls):
        """Gerekli dizinleri oluştur"""
        cls.PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

Config.setup_dirs()

class DocumentProcessor:
    """PDF belgelerini işleyen sınıf"""
    
    @staticmethod
    def download_pdf(url: str, save_path: Path) -> bool:
        """
        PDF'i indir ve kaydet
        
        Args:
            url: İndirilecek PDF URL'si
            save_path: Kaydedilecek dosya yolu
            
        Returns:
            bool: İndirme başarılıysa True, değilse False
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                f.write(response.content)
                
            logger.info(f"PDF başarıyla indirildi: {save_path}")
            return True
        except Exception as e:
            logger.error(f"PDF indirme hatası: {e}")
            return False
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        """
        PDF'den metin çıkar
        
        Args:
            pdf_path: PDF dosya yolu
            
        Returns:
            str: Çıkarılan metin
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
                
            return text
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Metni temizle ve normalize et
        
        Args:
            text: Temizlenecek metin
            
        Returns:
            str: Temizlenmiş metin
        """
        # Küçük harfe çevirme (embedding'ler case-sensitive olmayabilir)
        text = text.lower()
        
        # Özel karakterleri ve fazla boşlukları kaldır
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Stopword'leri kaldır
        stop_words = set(stopwords.words("english"))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return " ".join(words)
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE, 
                   chunk_overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
        """
        Metni chunk'lara ayır
        
        Args:
            text: Chunk'lanacak metin
            chunk_size: Her chunk'ın karakter boyutu
            chunk_overlap: Chunk'lar arası overlap
            
        Returns:
            List[str]: Chunk'lar listesi
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
                
            start = end - chunk_overlap
            
        return chunks
    
    @staticmethod
    def process_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
        """
        PDF'i işle ve chunk'ları hazırla
        
        Args:
            pdf_path: İşlenecek PDF dosya yolu
            
        Returns:
            List[Dict[str, Any]]: Chunk bilgileri listesi
        """
        try:
            # PDF'den metni çıkar
            raw_text = DocumentProcessor.extract_text_from_pdf(pdf_path)
            if not raw_text:
                return []
            
            # Metni temizle
            cleaned_text = DocumentProcessor.clean_text(raw_text)
            
            # Chunk'lara ayır
            chunks = DocumentProcessor.chunk_text(cleaned_text)
            
            # Metadata hazırla
            metadata = {
                "source": pdf_path.name,
                "year": DocumentProcessor.extract_year_from_filename(pdf_path.name),
                "processed_at": datetime.now().isoformat()
            }
            
            # Chunk'ları formatla
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{pdf_path.stem}_chunk_{i}"
                processed_chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": metadata
                })
            
            return processed_chunks
        except Exception as e:
            logger.error(f"PDF işleme hatası: {e}")
            return []
    
    @staticmethod
    def extract_year_from_filename(filename: str) -> Optional[int]:
        """
        Dosya adından yıl bilgisini çıkar
        
        Args:
            filename: Dosya adı
            
        Returns:
            Optional[int]: Bulunan yıl veya None
        """
        match = re.search(r"\d{4}", filename)
        return int(match.group()) if match else None

class VectorDatabase:
    """Vektör veritabanı işlemlerini yöneten sınıf"""
    
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(Config.VECTOR_DB_DIR)
        ))
        
        # Embedding fonksiyonu
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Koleksiyonu al veya oluştur
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Cosine benzerliği kullan
        )
        
        logger.info("Vektör veritabanı başlatıldı")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Belge chunk'larını vektör veritabanına ekle
        
        Args:
            documents: Eklenecek belgeler listesi
            
        Returns:
            bool: Ekleme başarılıysa True, değilse False
        """
        try:
            if not documents:
                return False
                
            # Belgeleri ayır
            ids = [doc["id"] for doc in documents]
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Vektör veritabanına ekle
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"{len(documents)} belge vektör veritabanına eklendi")
            return True
        except Exception as e:
            logger.error(f"Belge ekleme hatası: {e}")
            return False
    
    def query(self, query_text: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Vektör veritabanında sorgu yap
        
        Args:
            query_text: Sorgu metni
            top_k: Döndürülecek sonuç sayısı
            
        Returns:
            List[Dict[str, Any]]: Sorgu sonuçları
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları formatla
            formatted_results = []
            for doc, meta, dist in zip(results["documents"][0], 
                                       results["metadatas"][0], 
                                       results["distances"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "score": 1 - dist  # Cosine benzerliğine çevir
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Sorgu hatası: {e}")
            return []

class LLMService:
    """LLM ile etkileşimi yöneten sınıf"""
    
    def __init__(self):
        # Ollama LLM'sini başlat (yerel çalıştırıldığını varsayarak)
        self.llm = Ollama(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            num_ctx=Config.LLM_MAX_TOKENS
        )
        
        # Prompt şablonları
        self.qa_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Sen NTT DATA Business Solutions'ın sürdürülebilirlik raporları hakkında bilgi veren bir asistansın. 
            Aşağıdaki bağlamı kullanarak soruyu cevapla. Eğer cevabı bilmiyorsan "Bilmiyorum" de.
            
            Bağlam: {context}
            
            Sorumlulukların:
            - Sadece verilen bağlamdaki bilgilere dayanarak cevap ver
            - Yanlış veya tahmine dayalı bilgi verme
            - Cevaplarını net ve öz tut
            - Sürdürülebilirlik odaklı ol
            - Rakamlar ve verilerle destekle<|eot_id|>
            
            <|start_header_id|>user<|end_header_id|>
            Soru: {question}<|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["context", "question"]
        )
        
        self.hyde_prompt = PromptTemplate(
            template="""Aşağıdaki soruya muhtemel bir cevap yaz. Bu cevap, benzer belgeleri bulmak için kullanılacak.
            
            Soru: {question}
            
            Muhtemel Cevap:""",
            input_variables=["question"]
        )
        
        logger.info("LLM servisi başlatıldı")
    
    def generate_answer(self, context: str, question: str) -> str:
        """
        Bağlam ve soruya dayalı cevap oluştur
        
        Args:
            context: Cevap için bağlam
            question: Soru metni
            
        Returns:
            str: Oluşturulan cevap
        """
        try:
            # Prompt'u hazırla
            prompt = self.qa_prompt.format(context=context, question=question)
            
            # LLM'den cevap al
            response = self.llm(prompt)
            
            return response.strip()
        except Exception as e:
            logger.error(f"Cevap oluşturma hatası: {e}")
            return "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin."
    
    def generate_hyde_embedding(self, question: str) -> str:
        """
        Hypothetical Document Embedding (HyDE) oluştur
        
        Args:
            question: Soru metni
            
        Returns:
            str: HyDE dokümanı
        """
        try:
            prompt = self.hyde_prompt.format(question=question)
            hyde_doc = self.llm(prompt)
            return hyde_doc.strip()
        except Exception as e:
            logger.error(f"HyDE oluşturma hatası: {e}")
            return question  # Fallback olarak orijinal soruyu döndür

class RAGPipeline:
    """RAG pipeline'ını yöneten sınıf"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.llm_service = LLMService()
        
        # BM25 retriever için belgeleri sakla
        self.bm25_documents = []
        self.bm25_retriever = None
        
        logger.info("RAG pipeline başlatıldı")
    
    async def initialize_bm25_retriever(self):
        """BM25 retriever'ı başlat (asenkron)"""
        try:
            # Vektör veritabanındaki tüm belgeleri al
            all_docs = self.vector_db.collection.get(include=["documents", "metadatas"])
            texts = all_docs["documents"]
            metadatas = all_docs["metadatas"]
            
            # BM25 için belgeleri hazırla
            self.bm25_documents = [
                {"text": text, "metadata": meta} 
                for text, meta in zip(texts, metadatas)
            ]
            
            # BM25 retriever'ı oluştur
            if self.bm25_documents:
                texts_only = [doc["text"] for doc in self.bm25_documents]
                self.bm25_retriever = BM25Retriever.from_texts(
                    texts=texts_only,
                    metadatas=[doc["metadata"] for doc in self.bm25_documents]
                )
                self.bm25_retriever.k = Config.TOP_K_RETRIEVAL
                
                logger.info("BM25 retriever başlatıldı")
        except Exception as e:
            logger.error(f"BM25 retriever başlatma hatası: {e}")
    
    async def retrieve_documents(self, question: str) -> List[Dict[str, Any]]:
        """
        Belge alımı yap (gelişmiş RAG teknikleri ile)
        
        Args:
            question: Soru metni
            
        Returns:
            List[Dict[str, Any]]: Alınan belgeler
        """
        try:
            retrieval_methods = []
            
            # 1. Temel vektör benzerliği
            vector_results = self.vector_db.query(question)
            retrieval_methods.append(("vector", vector_results))
            
            # 2. HyDE (Hypothetical Document Embeddings)
            if Config.ENABLE_HYDE:
                hyde_doc = self.llm_service.generate_hyde_embedding(question)
                hyde_results = self.vector_db.query(hyde_doc)
                retrieval_methods.append(("hyde", hyde_results))
            
            # 3. Multi-query retrieval
            if Config.ENABLE_MULTI_QUERY and self.bm25_retriever:
                multi_query_results = await self._multi_query_retrieval(question)
                retrieval_methods.append(("multi_query", multi_query_results))
            
            # 4. BM25 (sparse retrieval)
            if self.bm25_retriever:
                bm25_results = self.bm25_retriever.get_relevant_documents(question)
                formatted_bm25 = [{
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.8  # BM25 için sabit skor (normalizasyon için)
                } for doc in bm25_results]
                retrieval_methods.append(("bm25", formatted_bm25))
            
            # Sonuçları birleştir ve yeniden sırala
            combined_results = await self._combine_and_rerank(retrieval_methods)
            
            # Yüksek puanlı sonuçları filtrele
            filtered_results = [
                res for res in combined_results 
                if res["score"] >= Config.CONFIDENCE_THRESHOLD
            ]
            
            return filtered_results[:Config.TOP_K_RETRIEVAL]
        except Exception as e:
            logger.error(f"Belge alım hatası: {e}")
            return []
    
    async def _multi_query_retrieval(self, question: str) -> List[Dict[str, Any]]:
        """
        Multi-query retrieval uygula
        
        Args:
            question: Orijinal soru
            
        Returns:
            List[Dict[str, Any]]: Alınan belgeler
        """
        try:
            # LangChain MultiQueryRetriever'ı kullan
            retriever = MultiQueryRetriever.from_llm(
                retriever=self.vector_db.collection.as_retriever(),
                llm=self.llm_service.llm
            )
            
            # Birden fazla sorgu oluştur ve çalıştır
            sub_questions = retriever.generate_queries(question, 3)
            all_results = []
            
            for q in sub_questions:
                results = self.vector_db.query(q)
                all_results.extend(results)
            
            # Tekilleştir ve sırala
            unique_results = {res["text"]: res for res in all_results}.values()
            sorted_results = sorted(unique_results, key=lambda x: x["score"], reverse=True)
            
            return sorted_results[:Config.TOP_K_RETRIEVAL]
        except Exception as e:
            logger.error(f"Multi-query retrieval hatası: {e}")
            return []
    
    async def _combine_and_rerank(self, retrieval_methods: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """
        Farklı retrieval yöntemlerinden gelen sonuçları birleştir ve yeniden sırala
        
        Args:
            retrieval_methods: (yöntem_adı, sonuçlar) listesi
            
        Returns:
            List[Dict[str, Any]]: Birleştirilmiş ve sıralanmış sonuçlar
        """
        try:
            combined = []
            
            # Tüm sonuçları topla
            for method_name, results in retrieval_methods:
                for res in results:
                    # Yönteme göre ağırlıklandırma
                    weight = 1.0
                    if method_name == "hyde":
                        weight = 0.9  # HyDE biraz daha az güvenilir
                    elif method_name == "bm25":
                        weight = 0.8  # BM25 daha az güvenilir
                    elif method_name == "multi_query":
                        weight = 1.1  # Multi-query daha güvenilir
                    
                    combined.append({
                        "text": res["text"],
                        "metadata": res["metadata"],
                        "score": res["score"] * weight,
                        "method": method_name
                    })
            
            # Tekilleştir
            unique_results = {}
            for res in combined:
                text = res["text"]
                if text not in unique_results or res["score"] > unique_results[text]["score"]:
                    unique_results[text] = res
            
            # Yeniden sırala
            sorted_results = sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)
            
            return sorted_results
        except Exception as e:
            logger.error(f"Sonuç birleştirme hatası: {e}")
            return []
    
    async def generate_response(self, question: str) -> Dict[str, Any]:
        """
        Soruya RAG tabanlı yanıt oluştur
        
        Args:
            question: Soru metni
            
        Returns:
            Dict[str, Any]: Yanıt ve kaynaklar
        """
        try:
            # 1. Belge alımı
            retrieved_docs = await self.retrieve_documents(question)
            
            if not retrieved_docs:
                return {
                    "answer": "İlgili bilgi bulunamadı.",
                    "sources": []
                }
            
            # 2. Bağlam oluştur
            context = "\n\n".join([doc["text"] for doc in retrieved_docs])
            
            # 3. LLM ile cevap oluştur
            answer = self.llm_service.generate_answer(context, question)
            
            # 4. Kaynakları hazırla
            sources = [
                {
                    "source": doc["metadata"]["source"],
                    "year": doc["metadata"]["year"],
                    "score": doc["score"],
                    "method": doc.get("method", "vector")
                }
                for doc in retrieved_docs
            ]
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Yanıt oluşturma hatası: {e}")
            return {
                "answer": "Üzgünüm, bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
                "sources": []
            }

class APIService:
    """FastAPI servisini yöneten sınıf"""
    
    def __init__(self):
        self.app = FastAPI(
            title="NTT DATA Sürdürülebilirlik Raporları API",
            description="NTT DATA'nın sürdürülebilirlik raporları üzerinde RAG tabanlı sorgulama yapar.",
            version="1.0.0"
        )
        
        # CORS ayarları
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # RAG pipeline'ı
        self.rag_pipeline = RAGPipeline()
        
        # API endpoint'lerini tanımla
        self._setup_routes()
        
        # Başlangıçta BM25 retriever'ı başlat
        asyncio.create_task(self.rag_pipeline.initialize_bm25_retriever())
        
        logger.info("API servisi başlatıldı")
    
    def _setup_routes(self):
        """API endpoint'lerini tanımla"""
        
        # Pydantic modelleri
        class QuestionRequest(BaseModel):
            question: str = Field(..., description="Sorulacak soru")
        
        class AnswerResponse(BaseModel):
            answer: str = Field(..., description="Oluşturulan cevap")
            sources: List[Dict[str, Any]] = Field(
                ..., 
                description="Cevabın kaynakları ve metadata bilgileri"
            )
        
        class HealthResponse(BaseModel):
            status: str = Field(..., description="Servis durumu")
            vector_db_count: int = Field(..., description="Vektör veritabanındaki belge sayısı")
            last_updated: Optional[str] = Field(None, description="Son güncelleme zamanı")
        
        # Endpoint'ler
        @self.app.post("/ask", response_model=AnswerResponse)
        async def ask_question(request: QuestionRequest):
            """Soruya cevap ver"""
            try:
                response = await self.rag_pipeline.generate_response(request.question)
                return response
            except Exception as e:
                logger.error(f"API hatası: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Servis sağlık durumunu kontrol et"""
            try:
                # Vektör veritabanı istatistikleri
                count = self.rag_pipeline.vector_db.collection.count()
                
                # Son güncelleme zamanı
                last_update = None
                if count > 0:
                    items = self.rag_pipeline.vector_db.collection.get(limit=1)
                    if items["metadatas"]:
                        last_update = items["metadatas"][0].get("processed_at")
                
                return {
                    "status": "healthy",
                    "vector_db_count": count,
                    "last_updated": last_update
                }
            except Exception as e:
                logger.error(f"Health check hatası: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Health check failed"
                )

class DataIngestion:
    """Veri yükleme işlemlerini yöneten sınıf"""
    
    @staticmethod
    def get_report_urls() -> List[Dict[str, str]]:
        """
        NTT DATA sürdürülebilirlik raporu URL'lerini al
        
        Returns:
            List[Dict[str, str]]: Rapor URL'leri ve metadata
        """
        # Örnek URL'ler - gerçek uygulamada web scraping yapılabilir
        return [
            {
                "url": "https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/sustainability/susatainability-report/2023/sr2023.pdf?rev=ae0d7ce3fce24daaae3616c47030b761",
                "year": 2023,
                "title": "NTT DATA Sustainability Report 2023"
            },
            {
                "url": "https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/sustainability/susatainability-report/2022/sr_2022.pdf?rev=d67a7088abe84e03af49b8f47d3cd31f",
                "year": 2022,
                "title": "NTT DATA Sustainability Report 2022"
            },
            {
                "url": "https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/sustainability/susatainability-report/2021/sr_2021.pdf?rev=aea7ae087b93439ea593a26c842b39dc",
                "year": 2021,
                "title": "NTT DATA Sustainability Report 2021"
            },
            {
                "url": "https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/sustainability/susatainability-report/2020/sr_2020.pdf?rev=4e3d9921539d48ee8bb109d5b4110dc5",
                "year": 2020,
                "title": "NTT DATA Sustainability Report 2020"
            },
            {
                "url": "https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/sustainability/susatainability-report/2019/sr_2019_p.pdf?rev=b4626c134ef348dd92a90ae9eb19249c",
                "year": 2019,
                "title": "NTT DATA Sustainability Report 2019"
            }
        ]
    
    @staticmethod
    def download_and_process_reports() -> bool:
        """
        Tüm raporları indir ve işle
        
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        try:
            report_urls = DataIngestion.get_report_urls()
            vector_db = VectorDatabase()
            
            # Paralel indirme ve işleme
            with ThreadPoolExecutor() as executor:
                futures = []
                
                for report in report_urls:
                    # PDF dosya adını oluştur
                    filename = f"ntt_sustainability_{report['year']}.pdf"
                    save_path = Config.PDF_STORAGE_DIR / filename
                    
                    # Eğer dosya zaten varsa atla
                    if save_path.exists():
                        logger.info(f"{filename} zaten var, atlanıyor")
                        continue
                    
                    # İndirme işlemini başlat
                    futures.append(executor.submit(
                        DataIngestion._download_and_process_single_report,
                        report["url"],
                        save_path,
                        vector_db
                    ))
                
                # Sonuçları bekle
                results = [f.result() for f in futures]
                
            return all(results)
        except Exception as e:
            logger.error(f"Rapor işleme hatası: {e}")
            return False
    
    @staticmethod
    def _download_and_process_single_report(url: str, save_path: Path, vector_db: VectorDatabase) -> bool:
        """
        Tek bir raporu indir ve işle
        
        Args:
            url: Rapor URL'si
            save_path: Kaydedilecek dosya yolu
            vector_db: Vektör veritabanı instance'ı
            
        Returns:
            bool: İşlem başarılıysa True, değilse False
        """
        try:
            # PDF'i indir
            if not DocumentProcessor.download_pdf(url, save_path):
                return False
            
            # PDF'i işle
            processed_chunks = DocumentProcessor.process_pdf(save_path)
            if not processed_chunks:
                return False
                
            # Vektör veritabanına ekle
            return vector_db.add_documents(processed_chunks)
        except Exception as e:
            logger.error(f"Tek rapor işleme hatası: {e}")
            return False

class GCPIntegration:
    """GCP entegrasyon işlemleri"""
    
    @staticmethod
    def upload_to_gcs(local_path: Path, gcs_path: str) -> bool:
        """
        Dosyayı Google Cloud Storage'a yükle
        
        Args:
            local_path: Yerel dosya yolu
            gcs_path: GCS hedef yolu
            
        Returns:
            bool: Yükleme başarılıysa True, değilse False
        """
        try:
            if not Config.GCP_BUCKET_NAME:
                logger.warning("GCP_BUCKET_NAME ayarlanmamış, atlanıyor")
                return False
                
            client = storage.Client()
            bucket = client.bucket(Config.GCP_BUCKET_NAME)
            blob = bucket.blob(gcs_path)
            
            blob.upload_from_filename(str(local_path))
            
            logger.info(f"{local_path} başarıyla GCS'ye yüklendi: gs://{Config.GCP_BUCKET_NAME}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"GCS yükleme hatası: {e}")
            return False
    
    @staticmethod
    def backup_vector_db() -> bool:
        """
        Vektör veritabanını GCS'ye yedekle
        
        Returns:
            bool: Yedekleme başarılıysa True, değilse False
        """
        try:
            if not Config.GCP_BUCKET_NAME:
                return False
                
            # Vektör veritabanı dosyalarını bul
            db_files = list(Config.VECTOR_DB_DIR.glob("*"))
            if not db_files:
                return False
                
            # Her dosyayı yükle
            success = True
            for db_file in db_files:
                gcs_path = f"vector_db/{db_file.name}"
                if not GCPIntegration.upload_to_gcs(db_file, gcs_path):
                    success = False
                    
            return success
        except Exception as e:
            logger.error(f"Vektör DB yedekleme hatası: {e}")
            return False

class Monitoring:
    """Sistem izleme ve metrikler"""
    
    @staticmethod
    def log_metrics(question: str, response: Dict[str, Any], response_time: float):
        """
        Sorgu metriklerini logla
        
        Args:
            question: Sorulan soru
            response: Alınan yanıt
            response_time: Yanıt süresi (saniye)
        """
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response_time": response_time,
                "answer_length": len(response.get("answer", "")),
                "sources_count": len(response.get("sources", [])),
                "avg_source_score": (
                    sum(s["score"] for s in response["sources"]) / len(response["sources"]) 
                    if response.get("sources") else 0
                ),
                "retrieval_methods": ",".join(
                    set(s.get("method", "vector") for s in response.get("sources", []))
            }
            
            # Log dosyasına yaz
            metrics_file = Config.PROCESSED_DIR / "metrics.log"
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
                
            # GCS'ye yedekle
            if Config.GCP_BUCKET_NAME:
                GCPIntegration.upload_to_gcs(
                    metrics_file,
                    f"metrics/metrics_{datetime.now().date()}.log"
                )
        except Exception as e:
            logger.error(f"Metrik loglama hatası: {e}")

def main():
    """Uygulama ana fonksiyonu"""
    try:
        # Veri yükleme işlemini başlat
        logger.info("Raporlar indiriliyor ve işleniyor...")
        DataIngestion.download_and_process_reports()
        
        # GCS yedekleme
        if Config.GCP_BUCKET_NAME:
            logger.info("Vektör veritabanı GCS'ye yedekleniyor...")
            GCPIntegration.backup_vector_db()
        
        # API servisini başlat
        api_service = APIService()
        
        logger.info(f"API servisi başlatılıyor: http://{Config.FASTAPI_HOST}:{Config.FASTAPI_PORT}")
        uvicorn.run(
            api_service.app,
            host=Config.FASTAPI_HOST,
            port=Config.FASTAPI_PORT,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Uygulama hatası: {e}")
        raise

if __name__ == "__main__":
    main()