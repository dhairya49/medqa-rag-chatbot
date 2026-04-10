"""
app/frontend/api_client.py

Thin HTTP client that wraps all FastAPI backend calls.
Uses synchronous httpx — Streamlit runs in a normal Python thread,
not an async event loop.

Endpoints used:
  GET  /api/v1/health       — health check
  POST /api/v1/chat         — RAG + drug tool (text only)
  POST /api/v1/chat/report  — report tool (PDF upload)
"""

import httpx
from dataclasses import dataclass, field

BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT  = 180.0   # Ollama can be slow locally


# ── Response models ───────────────────────────────────────────────────────────

@dataclass
class SourceChunk:
    chunk_text : str
    source     : str
    category   : str
    score      : float


@dataclass
class ChatResult:
    session_id : str
    answer     : str
    tool_used  : str | None
    sources    : list[SourceChunk] = field(default_factory=list)
    source_url : str | None = None
    error      : str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_sources(raw: list[dict]) -> list[SourceChunk]:
    chunks = []
    for item in raw:
        chunks.append(SourceChunk(
            chunk_text = item.get("chunk_text", ""),
            source     = item.get("source", ""),
            category   = item.get("category", ""),
            score      = float(item.get("score", 0.0)),
        ))
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def check_health() -> bool:
    """Returns True if backend is reachable and healthy."""
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def send_message(
    session_id : str,
    message    : str,
    mode       : str = "concise",
    top_k      : int = 5,
) -> ChatResult:
    """
    Send a text-only message to /chat.
    Handles RAG path and drug tool — agent decides which to use.
    """
    try:
        response = httpx.post(
            f"{BASE_URL}/chat",
            data={
                "session_id" : session_id,
                "message"    : message,
                "mode"       : mode,
                "top_k"      : top_k,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return ChatResult(
            session_id = data.get("session_id", session_id),
            answer     = data.get("answer", ""),
            tool_used  = data.get("tool_used"),
            sources    = _parse_sources(data.get("sources", [])),
            source_url = data.get("source_url"),
        )
    except httpx.HTTPStatusError as exc:
        return ChatResult(
            session_id=session_id,
            answer="",
            tool_used=None,
            error=f"Backend error {exc.response.status_code}: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        return ChatResult(
            session_id=session_id,
            answer="",
            tool_used=None,
            error=f"Could not reach backend: {exc}",
        )


def send_report(
    session_id : str,
    message    : str,
    pdf_bytes  : bytes,
    filename   : str,
    mode       : str = "detailed",
) -> ChatResult:
    """
    Send a PDF report to /chat/report.
    Always uses detailed mode by default — reports need full answers.
    """
    try:
        response = httpx.post(
            f"{BASE_URL}/chat/report",
            data={
                "session_id" : session_id,
                "message"    : message,
                "mode"       : mode,
            },
            files={
                "file": (filename, pdf_bytes, "application/pdf"),
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return ChatResult(
            session_id = data.get("session_id", session_id),
            answer     = data.get("answer", ""),
            tool_used  = data.get("tool_used", "report_tool"),
            sources    = _parse_sources(data.get("sources", [])),
        )
    except httpx.HTTPStatusError as exc:
        return ChatResult(
            session_id=session_id,
            answer="",
            tool_used=None,
            error=f"Backend error {exc.response.status_code}: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        return ChatResult(
            session_id=session_id,
            answer="",
            tool_used=None,
            error=f"Could not reach backend: {exc}",
        )