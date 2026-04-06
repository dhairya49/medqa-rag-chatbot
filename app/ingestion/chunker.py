"""
app/ingestion/chunker.py

Splits cleaned Q&A records into chunks suitable for embedding and retrieval.

Strategy:
  - Each Q&A pair is first combined: "Q: {question}\nA: {answer}"
  - The combined text is split into overlapping token windows.
  - Chunk size: CHUNK_SIZE tokens (default 400), overlap: CHUNK_OVERLAP (default 50).
  - Short texts that fit in one window are kept as a single chunk.
  - Each chunk carries the full metadata from its source record plus its own
    chunk_index and total_chunks fields.

Token counting uses tiktoken (cl100k_base encoding — fast, accurate for English).
We use token counts for splitting but store plain text (not token IDs) in Qdrant.
"""

from __future__ import annotations

from typing import Any

import tiktoken

from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Use the same encoding as GPT-4 / modern models — good proxy for most tokenisers
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return _ENCODING.encode(text)


def _detokenize(tokens: list[int]) -> str:
    return _ENCODING.decode(tokens)


def _split_into_windows(
    tokens: list[int],
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    """
    Slide a window of `chunk_size` tokens over the token list,
    stepping forward by (chunk_size - overlap) each time.
    Returns a list of token-id windows.
    """
    if len(tokens) <= chunk_size:
        return [tokens]

    step = chunk_size - overlap
    windows: list[list[int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        windows.append(tokens[start:end])
        if end == len(tokens):
            break
        start += step

    return windows


def chunk_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert cleaned Q&A records into a flat list of chunk dicts.

    Each chunk dict:
      {
          "text":          str,   # chunk text fed to the embedder
          "question":      str,   # original question (for metadata)
          "source":        str,
          "category":      str,
          "topic":         str,
          "chunk_index":   int,   # 0-based index within this record
          "total_chunks":  int,   # total chunks for this record
          "token_count":   int,   # actual token count of this chunk
      }
    """
    settings = get_settings()
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    all_chunks: list[dict[str, Any]] = []
    total_tokens = 0

    for record in records:
        # Combine Q and A into one text block for the chunk
        combined = f"Q: {record['question']}\nA: {record['answer']}"
        tokens = _tokenize(combined)
        windows = _split_into_windows(tokens, chunk_size, overlap)

        total_chunks = len(windows)
        for idx, window in enumerate(windows):
            chunk_text = _detokenize(window)
            token_count = len(window)
            total_tokens += token_count

            all_chunks.append(
                {
                    "text": chunk_text,
                    "question": record["question"],
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
        total_tokens=total_tokens,
        avg_tokens_per_chunk=round(avg_tokens, 1),
        chunk_size_setting=chunk_size,
        overlap_setting=overlap,
    )

    return all_chunks
