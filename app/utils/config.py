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
    qdrant_collection: str = "medquad_chunks"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 400
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5

    # ── LLM (Groq) ────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"   # Groq model name for Llama 3.1 8B
    llm_max_tokens: int = 2048

    # ── Dataset ───────────────────────────────────────────────────────────────
    hf_dataset_name: str = "lavita/medical-qa-datasets"
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
        print(settings.groq_api_key)
    """
    return Settings()




# """
# app/utils/config.py

# Centralised configuration using pydantic-settings.
# All values are read from environment variables or .env file.
# No hardcoding anywhere else in the codebase — import settings instead.
# """

# from functools import lru_cache
# from pydantic_settings import BaseSettings, SettingsConfigDict


# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         case_sensitive=False,
#         extra="ignore",
#     )

#     # ── Qdrant ────────────────────────────────────────────────────────────────
#     qdrant_host: str = "localhost"
#     qdrant_port: int = 6333
#     qdrant_collection: str = "medquad_chunks"

#     # ── Embedding ─────────────────────────────────────────────────────────────
#     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
#     embedding_dim: int = 384
#     embedding_batch_size: int = 64

#     # ── Chunking ──────────────────────────────────────────────────────────────
#     chunk_size: int = 400        # target tokens per chunk
#     chunk_overlap: int = 50      # overlap tokens between consecutive chunks

#     # ── Retrieval ─────────────────────────────────────────────────────────────
#     top_k: int = 5

#     # ── LLM ───────────────────────────────────────────────────────────────────
#     ollama_host: str = "http://localhost:11434"
#     llm_model: str = "llama3.1:8b"
#     llm_max_tokens: int = 2048

#     # ── Dataset ───────────────────────────────────────────────────────────────
#     hf_dataset_name: str = "lavita/medical-qa-datasets"
#     hf_dataset_split: str = "train"

#     # ── App ───────────────────────────────────────────────────────────────────
#     app_host: str = "0.0.0.0"
#     app_port: int = 8000
#     log_level: str = "INFO"


# @lru_cache(maxsize=1)
# def get_settings() -> Settings:
#     """
#     Return a cached singleton Settings instance.
#     Use this everywhere instead of instantiating Settings directly.

#     Example:
#         from app.utils.config import get_settings
#         settings = get_settings()
#         print(settings.qdrant_host)
#     """
#     return Settings()
