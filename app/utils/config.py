"""
app/utils/config.py

Centralised configuration using pydantic-settings.
All values are read from environment variables or .env file.
No hardcoding anywhere else in the codebase — import settings instead.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "https://928292ee-dad4-4b1b-b0c7-3b5f76308fa5.europe-west3-0.gcp.cloud.qdrant.io"
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZTBkMDJkMzctZWJiZC00ZjllLTg1NTUtMmZkZWI1YzQ3NWU2In0.PKZi5bypFPpTjQSQtw726N2ysfX95zI8tkOlF-5pOt4"
    qdrant_collection: str = "medquad_clean"          # new clean collection

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 350
    chunk_overlap: int = 75

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 8
    retrieval_candidate_pool: int = 24
    retrieval_keyword_weight: float = 0.35
    retrieval_dense_weight: float = 0.65

    # ── LLM (Mistral) ─────────────────────────────────────────────────────────
    mistral_api_key: str = ""
    llm_model: str = "mistral-small-latest"           # fast, free tier, no TPM limit
    llm_max_tokens: int = 1024                        # concise answers for speed 512
    llm_temperature: float = 0.15
    llm_top_p: float = 0.9

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_dataset_name: str = "lavita/MedQuAD"  # clean structured medical Q&A
    hf_dataset_split: str = "train"

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    redis_url: str = "redis://localhost:6379"
    session_ttl_seconds: int = 86400      # 24 hours
    session_max_turns: int = 10           # last 10 turns = 20 messages fetched

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.
    Use this everywhere instead of instantiating Settings directly.

    Example:
        from app.utils.config import get_settings
        settings = get_settings()
        print(settings.mistral_api_key)
    """
    return Settings()