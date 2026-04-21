"""
app/ingestion/loader.py

Loads the MedQuAD dataset from Hugging Face and returns a list of
raw Q&A records.
"""

from __future__ import annotations
from typing import Any
from datasets import load_dataset, Dataset
from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

_QUESTION_ALIASES = ["question", "input", "query", "q"]
_ANSWER_ALIASES = ["answer", "output", "response", "a"]
_SOURCE_ALIASES = ["source", "source_url", "url", "focus_area"]
_CATEGORY_ALIASES = ["category", "qtype", "question_type", "type"]
_TOPIC_ALIASES = ["topic", "condition", "disease", "focus"]


def _resolve(row: dict[str, Any], aliases: list[str], default: str = "") -> str:
    for key in aliases:
        val = row.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return default


def load_medquad() -> list[dict[str, Any]]:
    settings = get_settings()

    # Dataset config — 'all-processed' is the combined config for lavita/medical-qa-datasets
    dataset_name = settings.hf_dataset_name
    dataset_config = "default"
    dataset_split = settings.hf_dataset_split

    log.info("loading_dataset", dataset=dataset_name, config=dataset_config, split=dataset_split)

    try:
        ds: Dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
        )
    except Exception as exc:
        log.error("dataset_load_failed", error=str(exc))
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}'. "
            "Check your internet connection and the dataset name."
        ) from exc

    log.info("dataset_loaded", raw_rows=len(ds))

    records: list[dict[str, Any]] = []
    skipped = 0

    for row in ds:
        question = _resolve(row, _QUESTION_ALIASES)
        answer = _resolve(row, _ANSWER_ALIASES)

        if not question or not answer:
            skipped += 1
            continue

        records.append({
            "question": question,
            "answer": answer,
            "source": _resolve(row, _SOURCE_ALIASES, default="MedQuAD"),
            "category": _resolve(row, _CATEGORY_ALIASES, default="general"),
            "topic": _resolve(row, _TOPIC_ALIASES, default="unknown"),
        })

    log.info("loader_done", total_records=len(records), skipped_empty=skipped)
    return records