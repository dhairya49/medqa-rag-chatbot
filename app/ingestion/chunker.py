"""
app/ingestion/chunker.py

Splits cleaned Q&A records into chunks suitable for embedding and retrieval.

Strategy:
  - Each Q&A pair is first tried as a single chunk (Q&A pair preservation).
  - If it fits within CHUNK_SIZE tokens → kept as one chunk (ideal for retrieval).
  - If it exceeds CHUNK_SIZE → split using sentence-aware windowing:
      * Answer is split at clean sentence boundaries (no mid-sentence cuts).
      * Overlap is done by carrying over whole sentences (not raw tokens),
        preventing mangled/partial sentences at chunk boundaries.
  - Pure token-split fallback is used only when sentence splitting yields nothing.

Token counting uses tiktoken (cl100k_base encoding — fast, accurate for English).
We use token counts for splitting but store plain text (not token IDs) in Qdrant.

Changes from previous version:
  - Overlap is now whole-sentence based, not raw token slice based.
  - _detokenize(overlap_tokens) removed — was producing mangled partial sentences.
  - Cleaner sentence carry-over: last N sentences from previous window are prepended.
  - chunk_size default aligned with config (350 tokens, overlap 75).
"""

from __future__ import annotations

import re
from typing import Any

import tiktoken

from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# cl100k_base — same encoding as GPT-4, good proxy for most tokenisers
_ENCODING = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Low-level token helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[int]:
    return _ENCODING.encode(text)


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on '.', '!', '?' boundaries.
    Strips whitespace and drops empty fragments.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _sentence_aware_windows(
    question: str,
    answer: str,
    chunk_size: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Split a long Q&A pair into overlapping windows at sentence boundaries.

    Overlap strategy:
      - After filling a window, count back from the end of the sentence list
        until we accumulate >= overlap_tokens worth of text.
      - Those trailing sentences become the START of the next window.
      - This guarantees overlap is always whole, readable sentences — never
        a raw token slice that decodes to a mangled half-sentence.

    Args:
        question:       The original question string.
        answer:         The full answer string to be windowed.
        chunk_size:     Max tokens per chunk (including question prefix).
        overlap_tokens: Target token budget for sentence-level carry-over.

    Returns:
        List of chunk strings, each formatted as "Q: ...\nA: ..."
    """
    prefix = f"Q: {question}\nA: "
    prefix_len = _token_len(prefix)
    available = max(chunk_size - prefix_len, 80)  # tokens left for answer text

    sentences = _split_sentences(answer)
    if not sentences:
        # Degenerate case — return as single chunk even if oversized
        return [f"Q: {question}\nA: {answer}".strip()]

    windows: list[str] = []
    current_sentences: list[str] = []
    current_len: int = 0

    for sentence in sentences:
        s_len = _token_len(sentence)

        # If adding this sentence would overflow the window, flush first
        if current_sentences and current_len + s_len + 1 > available:
            # Build and store the completed window
            answer_text = " ".join(current_sentences)
            windows.append(f"{prefix}{answer_text}")

            # --- Whole-sentence overlap ---
            # Walk backwards through current_sentences, accumulating token
            # budget until we hit overlap_tokens. Those sentences carry over.
            carry: list[str] = []
            carry_len = 0
            for sent in reversed(current_sentences):
                sent_len = _token_len(sent)
                if carry_len + sent_len > overlap_tokens:
                    break
                carry.insert(0, sent)
                carry_len += sent_len

            # Start new window with carry-over sentences
            current_sentences = carry[:]
            current_len = carry_len

        current_sentences.append(sentence)
        current_len += s_len + 1  # +1 for the space separator

    # Flush any remaining sentences
    if current_sentences:
        answer_text = " ".join(current_sentences)
        windows.append(f"{prefix}{answer_text}")

    return windows


def _token_split_fallback(
    question: str,
    answer: str,
    chunk_size: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Last-resort fallback: pure token window split.
    Only reached when sentence splitting yields nothing (e.g. answer has no
    sentence-ending punctuation at all).
    """
    prefix = f"Q: {question}\nA: "
    prefix_ids = _tokenize(prefix)
    answer_ids = _tokenize(answer)
    available = max(chunk_size - len(prefix_ids), 80)

    if len(answer_ids) <= available:
        return [f"{prefix}{answer}"]

    step = max(available - overlap_tokens, 1)
    chunks: list[str] = []
    start = 0
    while start < len(answer_ids):
        end = min(start + available, len(answer_ids))
        chunk_ids = prefix_ids + answer_ids[start:end]
        chunks.append(_ENCODING.decode(chunk_ids))
        if end == len(answer_ids):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert cleaned Q&A records into a flat list of chunk dicts.

    Primary strategy:
      If the full "Q: ...\nA: ..." fits within chunk_size tokens → single chunk.

    Fallback strategy (long answers):
      Sentence-aware windowing with whole-sentence overlap.
      Pure token split only if sentence splitting yields nothing.

    Each chunk dict:
      {
          "text":          str,   # chunk text fed to the embedder
          "question":      str,   # original question (for metadata / reranking)
          "source":        str,
          "category":      str,
          "topic":         str,
          "chunk_index":   int,   # 0-based index within this record
          "total_chunks":  int,   # total chunks produced for this record
          "token_count":   int,   # actual token count of this chunk
      }
    """
    settings = get_settings()
    chunk_size = settings.chunk_size       # default 350
    overlap = settings.chunk_overlap       # default 75

    all_chunks: list[dict[str, Any]] = []
    total_tokens = 0
    single_chunk_records = 0
    multi_chunk_records = 0

    for record in records:
        question = record["question"]
        answer = record["answer"]
        combined = f"Q: {question}\nA: {answer}"

        # --- Primary: Q&A pair preservation ---
        if _token_len(combined) <= chunk_size:
            chunk_texts = [combined]
            single_chunk_records += 1
        else:
            # --- Fallback: sentence-aware windowing ---
            chunk_texts = _sentence_aware_windows(
                question=question,
                answer=answer,
                chunk_size=chunk_size,
                overlap_tokens=overlap,
            )
            # Last resort: pure token split
            if not chunk_texts:
                chunk_texts = _token_split_fallback(
                    question=question,
                    answer=answer,
                    chunk_size=chunk_size,
                    overlap_tokens=overlap,
                )
            multi_chunk_records += 1

        total_chunks = len(chunk_texts)
        for idx, chunk_text in enumerate(chunk_texts):
            token_count = _token_len(chunk_text)
            total_tokens += token_count

            all_chunks.append(
                {
                    "text": chunk_text,
                    "question": question,
                    "source": record["source"],
                    "category": record["category"],
                    "topic": record["topic"],
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "token_count": token_count,
                }
            )

    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0

    log.info(
        "chunker_done",
        input_records=len(records),
        output_chunks=len(all_chunks),
        single_chunk_records=single_chunk_records,
        multi_chunk_records=multi_chunk_records,
        total_tokens=total_tokens,
        avg_tokens_per_chunk=round(avg_tokens, 1),
        chunk_size_setting=chunk_size,
        overlap_setting=overlap,
    )

    return all_chunks