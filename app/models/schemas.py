"""
app/models/schemas.py

Pydantic models for all API request and response bodies.
These are the contracts between the client and the FastAPI layer.
No logic here — only shape definitions.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Sent by the client on every /chat call.

    session_id  : unique string per conversation (client generated)
                  used to identify the user — no server-side memory in Phase 3,
                  reserved for Phase 4 session persistence.
    message     : the user's text question or instruction.
    top_k       : optional override for how many chunks to retrieve from Qdrant.
                  if omitted, the value from config (default 5) is used.
    """
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override retrieval top-k")


class SourceChunk(BaseModel):
    """
    A single retrieved chunk returned alongside the answer.
    Lets the user see where the answer came from.
    """
    chunk_text: str = Field(..., description="The retrieved text passage")
    source: str = Field(..., description="Dataset source identifier")
    category: Optional[str] = Field(None, description="Medical category e.g. diseases, drugs")
    score: float = Field(..., description="Cosine similarity score (0-1)")


class ChatResponse(BaseModel):
    """
    Returned by /chat on every successful call.

    answer      : the LLM generated answer.
    sources     : list of Qdrant chunks used to ground the answer.
    tool_used   : name of the tool invoked if any (report_tool / drug_tool / None).
    session_id  : echoed back from the request.
    """
    session_id: str
    answer: str
    sources: list[SourceChunk] = []
    tool_used: Optional[str] = None


# ── Report Upload (Tool 1) ────────────────────────────────────────────────────

class ReportAnalysisResponse(BaseModel):
    """
    Returned when a PDF medical report is uploaded and analysed.
    Separate from ChatResponse to make the tool output explicit.
    """
    session_id: str
    explanation: str = Field(..., description="Plain-language explanation of the report")
    extracted_text_preview: str = Field(..., description="First 300 chars of extracted PDF text")
    sources: list[SourceChunk] = []
    tool_used: str = "report_tool"


# ── Drug Lookup (Tool 2) ──────────────────────────────────────────────────────

class DrugInfo(BaseModel):
    """
    Structured drug information scraped from drugs.com or FDA.
    """
    drug_name: str
    uses: Optional[str] = None
    side_effects: Optional[str] = None
    dosage: Optional[str] = None
    interactions: Optional[str] = None
    source_url: str = Field(..., description="URL the data was scraped from")


class DrugLookupResponse(BaseModel):
    """
    Returned when a drug name is detected and Tool 2 is invoked.
    """
    session_id: str
    answer: str = Field(..., description="LLM answer using scraped drug info as context")
    drug_info: Optional[DrugInfo] = None
    tool_used: str = "drug_tool"


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    Returned by GET /health.
    Checks all critical dependencies are reachable.
    """
    status: str = Field(..., description="ok or degraded")
    qdrant: str = Field(..., description="connected or unreachable")
    ollama: str = Field(..., description="connected or unreachable")
    embedding_model: str
    collection: str
    qdrant_point_count: Optional[int] = None