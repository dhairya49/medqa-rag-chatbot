"""
app/ingestion/indexer.py

Embeds chunks using Sentence Transformers and upserts them into Qdrant.

Key design decisions:
  - Deterministic point IDs: UUID5 derived from (source + chunk_index + question[:50]).
    This means re-running the pipeline overwrites existing points cleanly — no duplicates.
  - Batched embedding: chunks are embedded in batches of EMBEDDING_BATCH_SIZE (default 64).
  - Batched upsert: Qdrant upserts are also batched to avoid memory spikes.
  - Collection is created once with the correct config; if it already exists, it is reused.
  - Distance metric: Cosine (vectors are normalised before upload for efficiency).
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    OptimizersConfigDiff,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Namespace UUID for deterministic point ID generation (arbitrary fixed UUID)
_ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _make_point_id(source: str, question: str, chunk_index: int) -> str:
    """Generate a deterministic UUID5 for a chunk."""
    key = f"{source}::{question[:50]}::{chunk_index}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def _ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        log.info("collection_exists", collection=collection)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,   # build HNSW index after 20k vectors
        ),
    )
    log.info("collection_created", collection=collection, dim=dim)


def index_chunks(chunks: list[dict[str, Any]]) -> None:
    """
    Embed all chunks and upsert them into Qdrant.

    Args:
        chunks: Output from chunker.chunk_records()
    """
    settings = get_settings()

    # ── Load embedding model ──────────────────────────────────────────────────
    log.info("loading_embedding_model", model=settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    log.info("connecting_qdrant", host=settings.qdrant_host, port=settings.qdrant_port)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # ── Ensure collection exists ──────────────────────────────────────────────
    _ensure_collection(client, settings.qdrant_collection, settings.embedding_dim)

    # ── Embed and upsert in batches ───────────────────────────────────────────
    batch_size = settings.embedding_batch_size
    total = len(chunks)
    upserted = 0

    log.info("indexing_start", total_chunks=total, batch_size=batch_size)

    for batch_start in tqdm(range(0, total, batch_size), desc="Indexing"):
        batch = chunks[batch_start : batch_start + batch_size]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        # Embed — returns numpy array of shape (batch_size, dim)
        # normalize_embeddings=True → cosine similarity == dot product (faster at query time)
        vectors = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Build Qdrant PointStructs
        points: list[PointStruct] = []
        for chunk, vector in zip(batch, vectors):
            point_id = _make_point_id(
                chunk["source"], chunk["question"], chunk["chunk_index"]
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        "question": chunk["question"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "topic": chunk["topic"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                        "token_count": chunk["token_count"],
                    },
                )
            )

        # Upsert — overwrites existing points with same ID safely
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        upserted += len(points)

    # ── Final count verification ──────────────────────────────────────────────
    collection_info = client.get_collection(settings.qdrant_collection)
    stored_count = collection_info.points_count

    log.info(
        "indexing_done",
        chunks_upserted=upserted,
        qdrant_point_count=stored_count,
        collection=settings.qdrant_collection,
    )
