"""
app/services/embedding.py

Async-capable embedding service for inference time.
Runs CPU-bound encode() in a thread pool so it doesn't block the event loop.
"""

import asyncio
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("loading_embedding_model", model=settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)
        self._dim = settings.embedding_dim
        logger.info("embedding_model_ready", model=settings.embedding_model, dim=self._dim)

    def embed_query_sync(self, text: str) -> list[float]:
        """Synchronous encode — runs in thread pool executor."""
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """
        Async embed — offloads CPU-bound encoding to a thread pool.
        Does not block the event loop so other requests can proceed.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query_sync, text)

    @property
    def dim(self) -> int:
        return self._dim


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()