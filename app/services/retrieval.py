"""
app/services/retrieval.py

Async-capable Qdrant vector search service.
Runs blocking Qdrant calls in a thread pool.
"""

import asyncio
import re
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
        self._candidate_pool = settings.retrieval_candidate_pool
        self._dense_weight = settings.retrieval_dense_weight
        self._keyword_weight = settings.retrieval_keyword_weight

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

    def _keywords(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    def _rerank_chunks(
        self,
        query_text: str,
        chunks: list[SourceChunk],
    ) -> list[SourceChunk]:
        query_terms = self._keywords(query_text)
        reranked: list[tuple[float, SourceChunk]] = []
        for chunk in chunks:
            chunk_terms = self._keywords(chunk.chunk_text)
            if query_terms:
                keyword_overlap = len(query_terms & chunk_terms) / len(query_terms)
            else:
                keyword_overlap = 0.0
            combined_score = (
                (chunk.score * self._dense_weight)
                + (keyword_overlap * self._keyword_weight)
            )
            reranked.append((combined_score, chunk))

        reranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in reranked]

    def search_sync(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[SourceChunk]:
        """Synchronous Qdrant search — runs in thread pool executor."""
        logger.info("searching_qdrant", collection=self._collection, top_k=top_k)

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=max(top_k, self._candidate_pool),
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

        reranked = self._rerank_chunks(query_text, chunks)
        selected = reranked[:top_k]
        logger.info("qdrant_results", count=len(selected), candidates=len(chunks))
        return selected

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[SourceChunk]:
        """
        Async Qdrant search — offloads blocking network call to thread pool.
        """
        k = top_k or self._default_top_k
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_sync, query_text, query_vector, k)

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
