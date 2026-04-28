"""
Configuration module for RAG system.
Loads environment variables and provides configuration objects.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./rag.db"
    )
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_vision_model: str = os.getenv(
        "OPENAI_VISION_MODEL",
        os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "project_documents")
    
    # Embeddings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-base-en-v1.5"
    )
    embedding_batch_size: int = 32
    
    # RAG Settings
    top_k_documents: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_pdf_drawing_analysis: bool = os.getenv(
        "ENABLE_PDF_DRAWING_ANALYSIS",
        "true"
    ).lower() in {"1", "true", "yes", "on"}
    pdf_render_dpi: int = int(os.getenv("PDF_RENDER_DPI", "200"))
    max_page_image_bytes: int = int(os.getenv("MAX_PAGE_IMAGE_BYTES", str(8 * 1024 * 1024)))
    hybrid_candidate_multiplier: int = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4"))
    keyword_scan_limit: int = int(os.getenv("KEYWORD_SCAN_LIMIT", "500"))
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_file_types: list = ["pdf", "docx", "txt"]
    upload_dir: str = "uploads"
    
    # API
    api_title: str = "Document RAG API"
    api_version: str = "1.0.0"
    

settings = Settings()
