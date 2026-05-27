"""
app/ingestion/drug_ingestion.py

Bulk Drug Ingestion Pipeline — OpenFDA → Qdrant (drug_chunks collection)

Flow:
    top-100 drug list
    → RxNorm brand→generic resolver  (same as drug_tool.py)
    → OpenFDA label API              (indications, warnings, interactions …)
    → text chunker                   (respects settings.chunk_size / overlap)
    → all-MiniLM-L6-v2 embeddings   (reuses EmbeddingService)
    → upsert into Qdrant             (collection: drug_chunks, cosine)

Design decisions:
    - Separate collection (drug_chunks) — keeps drug vectors isolated from
      medquad_chunks so retrieval profiles can target them independently.
    - Idempotent — re-running skips drugs already present (checks by drug_name
      payload filter before upserting).  Pass --force to overwrite.
    - Rate-limited — 0.5 s delay between FDA calls to be a polite API citizen.
    - Chunking mirrors the main ingestion pipeline (chunk_size=350, overlap=75
      from config) so retrieval behaves consistently across both collections.
    - Progress is logged per drug so you can Ctrl-C and resume safely.

Usage:
    # normal run (skips already-ingested drugs)
    python -m app.ingestion.drug_ingestion

    # force re-ingest everything
    python -m app.ingestion.drug_ingestion --force

    # dry run — fetch & chunk without writing to Qdrant
    python -m app.ingestion.drug_ingestion --dry-run

    # ingest a custom subset (generic names, comma-separated)
    python -m app.ingestion.drug_ingestion --drugs "metformin,aspirin,ibuprofen"
"""

from __future__ import annotations

import argparse
import hashlib
import time
import uuid
from typing import Generator

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.services.embedding import get_embedding_service
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Constants ─────────────────────────────────────────────────────────────────

DRUG_COLLECTION   = "drug_chunks"          # separate from medquad_chunks
FDA_API_BASE      = "https://api.fda.gov/drug/label.json"
RXNORM_API_BASE   = "https://rxnav.nlm.nih.gov/REST"
REQUEST_TIMEOUT   = 12
INTER_DRUG_DELAY  = 0.5                    # seconds between FDA calls

# FDA label fields to extract (in priority order)
FDA_FIELDS = [
    "indications_and_usage",
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "contraindications",
    "dosage_and_administration",
    "mechanism_of_action",
    "pharmacodynamics",
    "pharmacokinetics",
    "overdosage",
    "use_in_specific_populations",
    "pregnancy",
    "nursing_mothers",
    "pediatric_use",
    "geriatric_use",
    "storage_and_handling",
]

# ── Top-100 most commonly prescribed drugs (US + India combined) ──────────────
# Generic names used directly — RxNorm resolution still runs to normalise spelling.

TOP_100_DRUGS: list[str] = [
    # Cardiovascular
    "atorvastatin", "rosuvastatin", "amlodipine", "lisinopril", "metoprolol",
    "carvedilol", "losartan", "valsartan", "bisoprolol", "ramipril",
    "enalapril", "furosemide", "spironolactone", "digoxin", "warfarin",
    "clopidogrel", "aspirin", "nitroglycerin", "isosorbide mononitrate",
    "hydralazine",
    # Diabetes
    "metformin", "glipizide", "glimepiride", "sitagliptin", "empagliflozin",
    "dapagliflozin", "insulin glargine", "insulin aspart", "pioglitazone",
    "liraglutide",
    # Respiratory
    "salbutamol", "salmeterol", "fluticasone", "budesonide", "montelukast",
    "ipratropium", "tiotropium", "theophylline", "prednisone",
    "methylprednisolone",
    # Pain / Anti-inflammatory
    "ibuprofen", "naproxen", "diclofenac", "acetaminophen", "tramadol",
    "morphine", "oxycodone", "gabapentin", "pregabalin", "celecoxib",
    # Antibiotics
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "metronidazole", "clarithromycin", "ceftriaxone", "levofloxacin",
    "trimethoprim", "nitrofurantoin",
    # Gastroenterology
    "omeprazole", "pantoprazole", "esomeprazole", "ranitidine",
    "domperidone", "ondansetron", "metoclopramide", "loperamide",
    "lactulose", "mesalazine",
    # Psychiatry / Neurology
    "sertraline", "fluoxetine", "escitalopram", "amitriptyline",
    "alprazolam", "clonazepam", "zolpidem", "quetiapine", "risperidone",
    "lithium",
    # Thyroid / Hormones
    "levothyroxine", "methimazole", "propylthiouracil",
    "testosterone", "estradiol",
    # Vitamins / Supplements (commonly Rx)
    "cholecalciferol", "ferrous sulfate", "folic acid", "cyanocobalamin",
    "calcium carbonate",
    # Anticoagulants / Antiplatelet
    "rivaroxaban", "apixaban", "dabigatran", "enoxaparin", "heparin",
    # Others
    "allopurinol", "colchicine", "hydroxychloroquine", "prednisolone",
    "dexamethasone", "loratadine", "cetirizine", "fexofenadine",
    "sildenafil", "tamsulosin",
]


# ── RxNorm resolver (mirrors drug_tool.py) ────────────────────────────────────

def _resolve_generic_name(drug_name: str) -> str:
    """Resolve brand/alternate names to generic via RxNorm. Returns original on failure."""
    try:
        r = httpx.get(
            f"{RXNORM_API_BASE}/rxcui.json",
            params={"name": drug_name, "search": 1},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code != 200:
            return drug_name

        rxcui_list = r.json().get("idGroup", {}).get("rxnormId", [])
        if not rxcui_list:
            return drug_name

        rxcui = rxcui_list[0]

        r2 = httpx.get(
            f"{RXNORM_API_BASE}/rxcui/{rxcui}/related.json",
            params={"tty": "IN"},
            timeout=REQUEST_TIMEOUT,
        )
        if r2.status_code != 200:
            return drug_name

        for group in r2.json().get("relatedGroup", {}).get("conceptGroup", []):
            props = group.get("conceptProperties", [])
            if props:
                return props[0].get("name", drug_name).lower()

        return drug_name

    except (httpx.RequestError, KeyError, ValueError) as exc:
        logger.warning("rxnorm_error", drug=drug_name, error=str(exc))
        return drug_name


# ── OpenFDA fetcher ───────────────────────────────────────────────────────────

def _fetch_fda_label(generic_name: str) -> dict[str, str] | None:
    """
    Fetch an FDA drug label for generic_name.

    Tries two search strategies:
      1. openfda.generic_name  (most precise for generics)
      2. openfda.brand_name    (fallback — catches brand-only entries)

    Returns a dict {field_name: text} or None if nothing found.
    """
    strategies = [
        f'openfda.generic_name:"{generic_name}"',
        f'openfda.brand_name:"{generic_name}"',
    ]

    for search_param in strategies:
        try:
            r = httpx.get(
                FDA_API_BASE,
                params={"search": search_param, "limit": 1},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                continue

            results = r.json().get("results", [])
            if not results:
                continue

            label   = results[0]
            openfda = label.get("openfda", {})
            sections: dict[str, str] = {}

            # Extract configured FDA fields
            for field in FDA_FIELDS:
                value = label.get(field)
                if value and isinstance(value, list) and value[0].strip():
                    sections[field] = value[0].strip()

            if not sections:
                continue

            # Enrich with openfda metadata
            sections["_meta_generic_name"] = ", ".join(
                openfda.get("generic_name", [generic_name])
            )
            sections["_meta_brand_names"] = ", ".join(
                openfda.get("brand_name", [])[:5]  # cap at 5 brands
            )
            sections["_meta_manufacturer"] = ", ".join(
                openfda.get("manufacturer_name", [])[:2]
            )
            sections["_meta_route"] = ", ".join(
                openfda.get("route", [])
            )
            sections["_source_search"] = search_param

            logger.info(
                "fda_fetched",
                drug=generic_name,
                strategy=search_param,
                fields=len(sections),
            )
            return sections

        except (httpx.RequestError, KeyError, ValueError) as exc:
            logger.warning("fda_fetch_error", drug=generic_name, error=str(exc))
            continue

    logger.warning("fda_no_results", drug=generic_name)
    return None


# ── Text chunker ──────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> Generator[str, None, None]:
    """
    Word-based sliding window chunker.
    Mirrors the main ingestion chunker so retrieval behaves consistently.
    """
    words = text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            yield chunk
        if end >= len(words):
            break
        start += chunk_size - overlap


def _label_to_chunks(
    generic_name: str,
    label_sections: dict[str, str],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Convert an FDA label dict into a list of chunk dicts ready for embedding.

    Each chunk carries:
        text          — the actual chunk text
        drug_name     — canonical generic name
        section       — FDA field the chunk came from
        brand_names   — pipe-separated brand names (for payload search)
        manufacturer  — manufacturer string
        route         — administration route
        source        — "openfda"
    """
    meta_generic  = label_sections.get("_meta_generic_name", generic_name)
    brand_names   = label_sections.get("_meta_brand_names", "")
    manufacturer  = label_sections.get("_meta_manufacturer", "")
    route         = label_sections.get("_meta_route", "")

    chunks = []
    for field in FDA_FIELDS:
        section_text = label_sections.get(field)
        if not section_text:
            continue

        # Prefix each chunk with drug name + section so the LLM always has context
        section_label = field.replace("_", " ").title()
        prefixed      = f"{meta_generic.title()} — {section_label}:\n{section_text}"

        for chunk_text in _chunk_text(prefixed, chunk_size, overlap):
            chunks.append({
                "text":         chunk_text,
                "drug_name":    generic_name,
                "section":      field,
                "brand_names":  brand_names,
                "manufacturer": manufacturer,
                "route":        route,
                "source":       "openfda",
            })

    return chunks


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _ensure_collection(client: QdrantClient, dim: int) -> None:
    """Create drug_chunks collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if DRUG_COLLECTION not in existing:
        client.create_collection(
            collection_name=DRUG_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("collection_created", name=DRUG_COLLECTION, dim=dim)
    else:
        logger.info("collection_exists", name=DRUG_COLLECTION)


def _drug_already_ingested(client: QdrantClient, drug_name: str) -> bool:
    """Return True if at least one point for this drug_name exists in drug_chunks."""
    results = client.scroll(
        collection_name=DRUG_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="drug_name", match=MatchValue(value=drug_name))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    points = results[0]
    return len(points) > 0


def _delete_drug(client: QdrantClient, drug_name: str) -> None:
    """Delete all points for a drug (used in --force mode)."""
    client.delete(
        collection_name=DRUG_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="drug_name", match=MatchValue(value=drug_name))]
        ),
    )
    logger.info("drug_deleted", drug=drug_name)


def _upsert_chunks(
    client: QdrantClient,
    chunks: list[dict],
    embedder,
    batch_size: int = 32,
) -> int:
    """Embed and upsert chunks into drug_chunks. Returns number of points inserted."""
    total  = 0
    points = []

    for chunk in chunks:
        vector = embedder.embed_query_sync(chunk["text"])
        # Deterministic UUID from (drug_name + section + text hash) for idempotency
        uid = str(uuid.UUID(
            hashlib.md5(
                f"{chunk['drug_name']}:{chunk['section']}:{chunk['text'][:80]}".encode()
            ).hexdigest()
        ))
        points.append(
            PointStruct(
                id      = uid,
                vector  = vector,
                payload = {
                    "text":         chunk["text"],
                    "drug_name":    chunk["drug_name"],
                    "section":      chunk["section"],
                    "brand_names":  chunk["brand_names"],
                    "manufacturer": chunk["manufacturer"],
                    "route":        chunk["route"],
                    "source":       chunk["source"],
                },
            )
        )

        if len(points) >= batch_size:
            client.upsert(collection_name=DRUG_COLLECTION, points=points)
            total  += len(points)
            points  = []

    if points:
        client.upsert(collection_name=DRUG_COLLECTION, points=points)
        total += len(points)

    return total


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ingestion(
    drug_list: list[str],
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Full ingestion pipeline.

    Args:
        drug_list: list of drug names (generic preferred, brand OK — RxNorm resolves)
        force:     if True, re-ingest drugs already present in Qdrant
        dry_run:   if True, fetch and chunk without writing to Qdrant
    """
    embedder = get_embedding_service()
    client   = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    if not dry_run:
        _ensure_collection(client, embedder.dim)

    chunk_size = settings.chunk_size   # 350
    overlap    = settings.chunk_overlap  # 75

    stats = {"total_drugs": len(drug_list), "ingested": 0, "skipped": 0,
             "failed": 0, "total_chunks": 0}

    logger.info(
        "ingestion_start",
        drugs=len(drug_list),
        collection=DRUG_COLLECTION,
        chunk_size=chunk_size,
        overlap=overlap,
        force=force,
        dry_run=dry_run,
    )

    for idx, raw_name in enumerate(drug_list, 1):
        raw_name = raw_name.strip().lower()
        logger.info("processing_drug", n=idx, total=len(drug_list), drug=raw_name)

        # ── 1. Resolve generic name ──────────────────────────────────────────
        generic_name = _resolve_generic_name(raw_name)
        if generic_name != raw_name:
            logger.info("name_resolved", original=raw_name, generic=generic_name)

        # ── 2. Skip if already ingested (unless --force) ─────────────────────
        if not dry_run and not force:
            if _drug_already_ingested(client, generic_name):
                logger.info("drug_skipped_exists", drug=generic_name)
                stats["skipped"] += 1
                continue

        if not dry_run and force:
            _delete_drug(client, generic_name)

        # ── 3. Fetch FDA label ────────────────────────────────────────────────
        label_sections = _fetch_fda_label(generic_name)
        if not label_sections:
            # Try original name if resolved name failed
            if generic_name != raw_name:
                logger.info("retrying_with_raw_name", drug=raw_name)
                label_sections = _fetch_fda_label(raw_name)

        if not label_sections:
            logger.warning("drug_failed_no_fda_data", drug=generic_name)
            stats["failed"] += 1
            time.sleep(INTER_DRUG_DELAY)
            continue

        # ── 4. Chunk ──────────────────────────────────────────────────────────
        chunks = _label_to_chunks(generic_name, label_sections, chunk_size, overlap)
        if not chunks:
            logger.warning("drug_no_chunks", drug=generic_name)
            stats["failed"] += 1
            time.sleep(INTER_DRUG_DELAY)
            continue

        logger.info("drug_chunked", drug=generic_name, chunks=len(chunks))

        # ── 5. Embed & upsert ─────────────────────────────────────────────────
        if not dry_run:
            inserted = _upsert_chunks(client, chunks, embedder)
            logger.info("drug_upserted", drug=generic_name, points=inserted)
            stats["total_chunks"] += inserted
        else:
            # Dry run: just show first chunk as a sanity check
            logger.info(
                "dry_run_sample",
                drug=generic_name,
                chunks=len(chunks),
                preview=chunks[0]["text"][:120],
            )
            stats["total_chunks"] += len(chunks)

        stats["ingested"] += 1
        time.sleep(INTER_DRUG_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "ingestion_complete",
        **stats,
        collection=DRUG_COLLECTION if not dry_run else "DRY_RUN",
    )
    print("\n" + "=" * 60)
    print(f"  Drug Ingestion Summary")
    print("=" * 60)
    print(f"  Total drugs       : {stats['total_drugs']}")
    print(f"  Ingested          : {stats['ingested']}")
    print(f"  Skipped (exists)  : {stats['skipped']}")
    print(f"  Failed (no data)  : {stats['failed']}")
    print(f"  Total chunks      : {stats['total_chunks']}")
    print(f"  Collection        : {DRUG_COLLECTION if not dry_run else 'DRY RUN'}")
    print("=" * 60 + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest top-100 drug data from OpenFDA into Qdrant (drug_chunks)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest drugs even if already present in Qdrant",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and chunk without writing to Qdrant",
    )
    parser.add_argument(
        "--drugs",
        type=str,
        default=None,
        help="Comma-separated list of drug names to ingest (overrides TOP_100 list)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    drug_list = (
        [d.strip() for d in args.drugs.split(",") if d.strip()]
        if args.drugs
        else TOP_100_DRUGS
    )

    run_ingestion(
        drug_list=drug_list,
        force=args.force,
        dry_run=args.dry_run,
    )