"""
app/services/retrieval.py

Async-capable Qdrant vector search service.
Runs blocking Qdrant calls in a thread pool.
"""

import asyncio
from functools import lru_cache

from qdrant_client import QdrantClient

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.models.schemas import SourceChunk

logger = get_logger(__name__)


class RetrievalService:

    def __init__(self) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._default_top_k = settings.top_k

        logger.info(
            "connecting_qdrant",
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=self._collection,
        )
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            check_compatibility=False,
        )
        logger.info("qdrant_connected")

    def search_sync(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[SourceChunk]:
        """Synchronous Qdrant search — runs in thread pool executor."""
        logger.info("searching_qdrant", collection=self._collection, top_k=top_k)

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks: list[SourceChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunks.append(
                SourceChunk(
                    chunk_text=payload.get("text", ""),
                    source=payload.get("source", "MedQuAD"),
                    category=payload.get("category", "general"),
                    topic=payload.get("topic", "unknown"),
                    score=point.score,
                )
            )

        logger.info("qdrant_results", count=len(chunks))
        return chunks

    async def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[SourceChunk]:
        """
        Async Qdrant search — offloads blocking network call to thread pool.
        """
        k = top_k or self._default_top_k
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_sync, query_vector, k)

    def health_check_sync(self) -> tuple[bool, int]:
        try:
            info = self._client.get_collection(self._collection)
            return True, info.points_count
        except Exception:
            return False, 0

    async def health_check(self) -> tuple[bool, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.health_check_sync)


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()