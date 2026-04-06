"""
app/ingestion/cleaner.py

Text cleaning pipeline for MedQuAD records.

Cleaning steps (applied to both question and answer):
  1. Decode HTML entities
  2. Strip HTML tags
  3. Normalise unicode (NFKC)
  4. Collapse whitespace (spaces, tabs, multiple newlines)
  5. Remove control characters
  6. Strip leading/trailing whitespace

Additionally:
  - Records with answer length < MIN_ANSWER_CHARS are dropped (too short to be useful).
  - Duplicate records (same question text) are deduplicated — first occurrence wins.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from app.utils.logger import get_logger

log = get_logger(__name__)

# Minimum character length for an answer to be kept
MIN_ANSWER_CHARS = 30

# Regex: strip HTML tags
_RE_HTML_TAG = re.compile(r"<[^>]+>")

# Regex: collapse 2+ whitespace/newlines into one space
_RE_WHITESPACE = re.compile(r"[\s\u00a0]+")

# Regex: remove ASCII control characters (except newline/tab which whitespace handles)
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def _clean_text(text: str) -> str:
    """Apply all cleaning steps to a single string."""
    # 1. Decode HTML entities (e.g., &amp; → &, &#39; → ')
    text = html.unescape(text)
    # 2. Strip HTML tags
    text = _RE_HTML_TAG.sub(" ", text)
    # 3. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)
    # 4. Remove control characters
    text = _RE_CONTROL.sub("", text)
    # 5. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text)
    # 6. Strip
    return text.strip()


def clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Clean a list of raw Q&A records.

    Args:
        records: Output from loader.load_medquad()

    Returns:
        Cleaned records with duplicates and short answers removed.
    """
    seen_questions: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    dropped_short = 0
    dropped_dupe = 0

    for record in records:
        question = _clean_text(record.get("question", ""))
        answer = _clean_text(record.get("answer", ""))

        # Drop if answer too short to be useful
        if len(answer) < MIN_ANSWER_CHARS:
            dropped_short += 1
            continue

        # Deduplicate on cleaned question text
        q_key = question.lower()
        if q_key in seen_questions:
            dropped_dupe += 1
            continue
        seen_questions.add(q_key)

        cleaned.append(
            {
                "question": question,
                "answer": answer,
                "source": _clean_text(record.get("source", "MedQuAD")),
                "category": _clean_text(record.get("category", "general")),
                "topic": _clean_text(record.get("topic", "unknown")),
            }
        )

    log.info(
        "cleaner_done",
        input_count=len(records),
        output_count=len(cleaned),
        dropped_short=dropped_short,
        dropped_duplicates=dropped_dupe,
    )
    return cleaned
