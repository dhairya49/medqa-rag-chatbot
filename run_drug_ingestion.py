"""
run_drug_ingestion.py

Entry point for the offline drug corpus ingestion pipeline.
"""

import sys
import time

from app.ingestion.drug_loader import load_drug_chunks
from app.ingestion.indexer import index_chunks
from app.utils.config import get_settings
from app.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger("run_drug_ingestion")


def main() -> None:
    settings = get_settings()
    start = time.perf_counter()

    log.info(
        "drug_pipeline_start",
        collection=settings.qdrant_drug_collection,
        limit=settings.drug_ingestion_limit or "all",
    )

    try:
        chunks = load_drug_chunks()
    except Exception as exc:
        log.error("drug_pipeline_failed_at_load", error=str(exc))
        sys.exit(1)

    if not chunks:
        log.error("drug_pipeline_failed", reason="No drug chunks were produced")
        sys.exit(1)

    try:
        index_chunks(chunks, collection_name=settings.qdrant_drug_collection)
    except Exception as exc:
        log.error("drug_pipeline_failed_at_indexing", error=str(exc))
        sys.exit(1)

    elapsed = time.perf_counter() - start
    log.info(
        "drug_pipeline_complete",
        total_chunks_indexed=len(chunks),
        elapsed_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    main()
