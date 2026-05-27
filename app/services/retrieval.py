"""
app/services/retrieval.py

Async-capable Qdrant vector search service.
Runs blocking Qdrant calls in a thread pool.

Collections:
  medquad_chunks  — general medical Q&A (existing)
  drug_chunks     — OpenFDA drug labels  (new, ingested via drug_ingestion.py)
"""

import asyncio
import re
from functools import lru_cache

from qdrant_client import QdrantClient

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.models.schemas import SourceChunk

logger = get_logger(__name__)

DRUG_COLLECTION = "drug_chunks"


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
            url=settings.qdrant_url,
            collection=self._collection,
        )
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            check_compatibility=False,
        )
        logger.info("qdrant_connected")

    # ── Shared helpers ────────────────────────────────────────────────────────

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

    # ── medquad_chunks search (existing) ─────────────────────────────────────

    def search_sync(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[SourceChunk]:
        """Synchronous Qdrant search over medquad_chunks — runs in thread pool."""
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
        """Async search over medquad_chunks — offloads to thread pool."""
        k = top_k or self._default_top_k
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_sync, query_text, query_vector, k)

    # ── drug_chunks search (new) ──────────────────────────────────────────────

    def search_drug_chunks_sync(
        self,
        drug_name: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 8,
    ) -> list[SourceChunk]:
        """
        Synchronous search over drug_chunks collection.

        Filters by drug_name payload first so results are always
        scoped to the requested drug, then reranks by dense + keyword score.

        Args:
            drug_name:    resolved generic name (e.g. "metformin")
            query_text:   original user question (for keyword rerank)
            query_vector: embedded query vector
            top_k:        number of chunks to return after rerank

        Returns:
            list[SourceChunk] — empty list if drug not in collection
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        logger.info(
            "searching_drug_chunks",
            drug=drug_name,
            top_k=top_k,
        )

        # candidate pool: fetch more than top_k so reranker has room to work
        candidate_limit = max(top_k, self._candidate_pool)

        try:
            results = self._client.query_points(
                collection_name=DRUG_COLLECTION,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="drug_name",
                            match=MatchValue(value=drug_name),
                        )
                    ]
                ),
                limit=candidate_limit,
                with_payload=True,
            )
        except Exception as exc:
            # Collection may not exist yet or Qdrant is unreachable
            logger.warning("drug_chunks_search_failed", drug=drug_name, error=str(exc))
            return []

        if not results.points:
            logger.info("drug_chunks_no_results", drug=drug_name)
            return []

        chunks: list[SourceChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunks.append(
                SourceChunk(
                    chunk_text=payload.get("text", ""),
                    source="OpenFDA",
                    # reuse category/topic fields to carry drug metadata
                    category=payload.get("section", "drug_info"),
                    topic=payload.get("drug_name", drug_name),
                    score=point.score,
                )
            )

        reranked = self._rerank_chunks(query_text, chunks)
        selected = reranked[:top_k]

        logger.info(
            "drug_chunks_results",
            drug=drug_name,
            candidates=len(chunks),
            returned=len(selected),
        )
        return selected

    async def search_drug_chunks(
        self,
        drug_name: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 8,
    ) -> list[SourceChunk]:
        """
        Async wrapper for search_drug_chunks_sync.
        Called from agent.py drug route — offloads blocking Qdrant call
        to thread pool so the event loop stays free.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search_drug_chunks_sync,
            drug_name,
            query_text,
            query_vector,
            top_k,
        )

    # ── Health checks ─────────────────────────────────────────────────────────

    def health_check_sync(self) -> tuple[bool, int]:
        try:
            info = self._client.get_collection(self._collection)
            return True, info.points_count
        except Exception:
            return False, 0

    async def health_check(self) -> tuple[bool, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.health_check_sync)

    def drug_collection_health_sync(self) -> tuple[bool, int]:
        """Check drug_chunks collection exists and return point count."""
        try:
            info = self._client.get_collection(DRUG_COLLECTION)
            return True, info.points_count
        except Exception:
            return False, 0

    async def drug_collection_health(self) -> tuple[bool, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.drug_collection_health_sync)


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()