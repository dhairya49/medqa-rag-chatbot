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
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
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