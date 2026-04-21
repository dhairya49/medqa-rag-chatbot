"""
run_ingestion.py

Entry point for the MedQuAD ingestion pipeline.

Runs the full pipeline:
  Load → Clean → Chunk → Embed → Store in Qdrant

Usage:
  python run_ingestion.py

The script is idempotent — re-running it overwrites existing Qdrant points
with the same deterministic IDs. No duplicates are created.

Environment:
  Copy .env.example to .env and fill in values before running.
  Qdrant must be running (docker-compose up qdrant).
"""

import sys
import time

from app.utils.config import get_settings
from app.utils.logger import get_logger, setup_logging
from app.ingestion.loader import load_medquad
from app.ingestion.cleaner import clean_records
from app.ingestion.chunker import chunk_records
from app.ingestion.indexer import index_chunks

setup_logging()
log = get_logger("run_ingestion")


def main() -> None:
    get_settings.cache_clear()

    settings = get_settings()
    start = time.perf_counter()

    log.info(
        "pipeline_start",
        dataset=settings.hf_dataset_name,
        qdrant_host=settings.qdrant_host,
        collection=settings.qdrant_collection,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    try:
        raw_records = load_medquad()
    except RuntimeError as exc:
        log.error("pipeline_failed_at_load", error=str(exc))
        sys.exit(1)

    if not raw_records:
        log.error("pipeline_failed", reason="No records returned from loader")
        sys.exit(1)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    clean = clean_records(raw_records)

    if not clean:
        log.error("pipeline_failed", reason="No records survived cleaning")
        sys.exit(1)

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    chunks = chunk_records(clean)

    if not chunks:
        log.error("pipeline_failed", reason="No chunks produced")
        sys.exit(1)

    # ── Step 4 + 5: Embed & Store ─────────────────────────────────────────────
    try:
        index_chunks(chunks)
    except Exception as exc:
        log.error("pipeline_failed_at_indexing", error=str(exc))
        sys.exit(1)

    elapsed = time.perf_counter() - start
    log.info(
        "pipeline_complete",
        total_chunks_indexed=len(chunks),
        elapsed_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    main()
