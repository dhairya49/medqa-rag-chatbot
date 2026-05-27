"""
app/services/agent.py

Async RAG agent — all blocking calls run in thread pools.

Session memory (new):
  - SessionMemory is held as an internal service on RAGAgent.
  - On every /chat request: history is fetched from Redis before the LLM call,
    and the completed turn is appended after.
  - If Redis is unavailable, history degrades to empty list — chat still works.
  - Report tool path skips history (PDF context is self-contained).

Drug routing:
  1. Detect drug name via regex
  2. Resolve brand → generic via RxNorm
  3. Search drug_chunks in Qdrant
  4. If chunks found → RAG answer with DRUG_PROMPT
  5. If no chunks   → fallback to drug_tool.lookup_drug() (live FDA fetch)
"""

import asyncio
import re
from app.utils.logger import get_logger
from app.services.embedding import get_embedding_service
from app.services.retrieval import get_retrieval_service
from app.services.llm import get_llm_service
from app.services.session import SessionMemory
from app.models.schemas import ChatResponse, SourceChunk

logger = get_logger(__name__)

# ── Prompt templates (RAG general) ───────────────────────────────────────────

PROMPT_TEMPLATES = {
    "concise": """\
You are a careful medical RAG assistant. Answer the user's question using only the supplied context.

Rules:
- Give a short answer in 2-4 bullet points.
- Include only facts supported by the context.
- If the context is incomplete, explicitly say what is missing.
- Do not speculate, diagnose, or invent treatments.
- If symptoms or urgent concerns are mentioned, advise consulting a healthcare professional.

Context:
{context}

{history}Question:
{question}

Answer:""",
    "detailed": """\
You are a careful medical RAG assistant. Produce a complete, grounded answer using only the supplied context.

Rules:
- Cover all important points from the context that answer the question.
- Prefer precise medical wording, then explain it in plain language.
- Do not add unsupported facts or generic filler.
- If relevant information is missing, state that clearly.
- End with a brief caution to consult a healthcare professional for diagnosis or treatment decisions.

Required format:
Summary:
- ...

Key Details:
- ...

Context:
{context}

{history}Question:
{question}

Response:""",
    "structured": """\
You are a careful medical RAG assistant. Build a structured answer using only the supplied context.

Rules:
- Include every relevant point supported by the context.
- Do not add information that is not explicitly supported.
- Use short bullet lists and concrete wording.
- If a section is not supported by the context, write "Not stated in the provided context."
- Finish with a brief note recommending professional medical advice when appropriate.

Required sections:
Direct Answer:
- ...

Symptoms / Signs:
- ...

Causes / Risk Factors:
- ...

Diagnosis / Tests:
- ...

Treatment / Management:
- ...

Context:
{context}

{history}Question:
{question}

Structured answer:""",
}

# ── Drug-specific RAG prompt ──────────────────────────────────────────────────

DRUG_RAG_PROMPT = """\
You are a careful medical assistant answering drug-related questions using only verified FDA label data.

Rules:
- Use only the information supplied in the context below.
- Use short bullet points and plain language.
- If the context does not cover the user's exact question, say so clearly.
- Do not recommend specific dosages — describe only what the source states.
- Always remind the user to consult their doctor or pharmacist before taking any medication.

Required format:
Direct Answer:
- ...

Safety Notes:
- ...

Source-Supported Details:
- ...

Context (FDA label data):
{context}

{history}User question:
{question}

Answer:"""

# ── History formatter ─────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """
    Convert a list of {role, content} messages into a plain-text block
    that slots into the {history} placeholder in every prompt template.

    Returns an empty string (not a labelled block) when history is empty
    so prompts render cleanly for first-turn requests.
    """
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    lines.append("")   # blank line before the current question
    return "\n".join(lines) + "\n"

# ── Drug detection ────────────────────────────────────────────────────────────

_DRUG_PATTERNS = [
    r"\b(?:drug|medication|medicine)[:\s]+([A-Za-z][A-Za-z0-9\-]{3,})",
    r"\bside[\s\-]?effects\s+of\s+([A-Za-z][A-Za-z0-9\-]{3,})",
    r"\bdosage\s+of\s+([A-Za-z][A-Za-z0-9\-]{3,})",
    r"\bdose\s+of\s+([A-Za-z][A-Za-z0-9\-]{3,})",
    r"\binteractions?\s+(?:for|of|with)\s+([A-Za-z][A-Za-z0-9\-]{3,})",
    r"\bhow\s+(?:to|do\s+I)\s+take\s+([A-Za-z][A-Za-z0-9\-]{3,})",
    r"\b(?:is|can\s+I\s+take)\s+([A-Za-z][A-Za-z0-9\-]{3,})\s+safe",
    r"\bcan\s+I\s+take\s+([A-Za-z][A-Za-z0-9\-]{3,})\b",
    r"\b([A-Za-z][A-Za-z0-9\-]{3,})\s+(?:dosage|dose|side[\s\-]?effects|interactions?|overdose|warnings?)\b",
]

_DRUG_EXCLUSIONS = {
    "diabetes", "cancer", "fever", "pain", "cold", "flu", "cough",
    "infection", "virus", "bacteria", "disease", "disorder", "syndrome",
    "condition", "treatment", "therapy", "surgery", "doctor", "patient",
    "hospital", "blood", "heart", "lung", "liver", "kidney", "brain",
    "chest", "back", "head", "neck", "skin", "bone", "joint", "muscle",
    "weight", "diet", "food", "water", "sleep", "stress", "anxiety",
    "depression", "health", "body", "immune", "allergy", "allergic",
    "asthma", "stroke", "attack", "pressure", "sugar", "insulin",
    "symptoms", "symptom", "diagnosis", "causes", "cause", "prevent",
    "prevention", "risk", "risks", "test", "tests", "levels", "normal",
}


def _detect_drug_name(message: str) -> str | None:
    for pattern in _DRUG_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().lower()
            if candidate not in _DRUG_EXCLUSIONS:
                return match.group(1).strip()
    return None


# ── Agent ─────────────────────────────────────────────────────────────────────

class RAGAgent:

    def __init__(self) -> None:
        self._embedder  = get_embedding_service()
        self._retriever = get_retrieval_service()
        self._llm       = get_llm_service()
        self._memory    = SessionMemory()          # Redis-backed, degrades gracefully
        logger.info("rag_agent_initialised")

    async def run(
        self,
        session_id: str,
        message: str,
        mode: str = "concise",
        top_k: int | None = None,
        pdf_bytes: bytes | None = None,
    ) -> ChatResponse:
        """
        Async main entry point for every /chat request.

        Routing order:
          1. PDF uploaded       → report_tool  (no history — PDF is self-contained)
          2. Drug name detected → drug_chunks RAG  (fallback: drug_tool)
          3. Everything else    → medquad_chunks RAG
        """

        # ── Route 1: PDF uploaded → report_tool (history skipped intentionally) ──
        if pdf_bytes is not None:
            logger.info("routing_to_report_tool", session_id=session_id)
            from app.tools.report_tool import analyse_report
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: analyse_report(
                    pdf_bytes=pdf_bytes,
                    user_question=message,
                    retriever=self._retriever,
                    embedder=self._embedder,
                    llm=self._llm,
                )
            )
            return ChatResponse(
                session_id=session_id,
                answer=result["answer"],
                sources=result["sources"],
                tool_used="report_tool",
            )

        # ── Fetch history (shared by both remaining routes) ───────────────────
        history = await self._memory.get_history(session_id)

        # ── Route 2: Drug detected → drug_chunks RAG + fallback ──────────────
        drug_name = _detect_drug_name(message)
        if drug_name:
            response = await self._drug_rag_answer(
                session_id=session_id,
                message=message,
                raw_drug_name=drug_name,
                top_k=top_k,
                history=history,
            )
            await self._memory.append_turn(session_id, message, response.answer)
            return response

        # ── Route 3: General Q&A → medquad_chunks RAG ────────────────────────
        logger.info("routing_to_rag", session_id=session_id)
        response = await self._rag_answer(session_id, message, mode, top_k, history)
        await self._memory.append_turn(session_id, message, response.answer)
        return response

    # ── Drug RAG path ─────────────────────────────────────────────────────────

    async def _drug_rag_answer(
        self,
        session_id: str,
        message: str,
        raw_drug_name: str,
        top_k: int | None,
        history: list[dict],
    ) -> ChatResponse:
        from app.tools.drug_tool import _resolve_generic_name
        loop = asyncio.get_event_loop()
        resolved_name = await loop.run_in_executor(
            None, _resolve_generic_name, raw_drug_name
        )
        logger.info(
            "routing_to_drug_rag",
            session_id=session_id,
            raw=raw_drug_name,
            resolved=resolved_name,
        )

        query_vector = await self._embedder.embed_query(message)
        k = top_k or 8
        chunks: list[SourceChunk] = await self._retriever.search_drug_chunks(
            drug_name=resolved_name,
            query_text=message,
            query_vector=query_vector,
            top_k=k,
        )

        if chunks:
            logger.info(
                "drug_rag_chunks_found",
                session_id=session_id,
                drug=resolved_name,
                chunks=len(chunks),
            )
            context = "\n\n".join(
                f"[{c.category.replace('_', ' ').title()}]\n{c.chunk_text}"
                for c in chunks
            )
            history_block = _format_history(history)
            prompt = DRUG_RAG_PROMPT.format(
                context=context,
                history=history_block,
                question=message,
            )
            answer = await self._llm.invoke(prompt)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                sources=chunks,
                tool_used="drug_rag",
            )

        # No chunks → live fallback (history not injected, tool handles its own prompt)
        logger.info(
            "drug_rag_no_chunks_fallback",
            session_id=session_id,
            drug=resolved_name,
        )
        from app.tools.drug_tool import lookup_drug
        result = await loop.run_in_executor(
            None,
            lambda: lookup_drug(
                drug_name=raw_drug_name,
                user_question=message,
                llm=self._llm,
            )
        )
        return ChatResponse(
            session_id=session_id,
            answer=result["answer"],
            sources=[],
            tool_used="drug_tool_fallback",
        )

    # ── General RAG path ──────────────────────────────────────────────────────

    async def _rag_answer(
        self,
        session_id: str,
        message: str,
        mode: str,
        top_k: int | None,
        history: list[dict],
    ) -> ChatResponse:
        query_vector = await self._embedder.embed_query(message)
        chunks: list[SourceChunk] = await self._retriever.search(
            query_text=message,
            query_vector=query_vector,
            top_k=top_k,
        )

        if not chunks:
            logger.warning("no_chunks_retrieved", session_id=session_id)
            no_result_answer = (
                "I could not find relevant medical information for your question. "
                "Please consult a healthcare professional."
            )
            # Still append so follow-up questions have context
            await self._memory.append_turn(session_id, message, no_result_answer)
            return ChatResponse(
                session_id=session_id,
                answer=no_result_answer,
                sources=[],
                tool_used=None,
            )

        context = "\n\n".join(
            f"[Source: {c.source}]\n{c.chunk_text}" for c in chunks
        )
        history_block = _format_history(history)
        prompt_template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["concise"])
        prompt = prompt_template.format(
            context=context,
            history=history_block,
            question=message,
        )
        answer = await self._llm.invoke(prompt)

        logger.info(
            "rag_answer_done",
            session_id=session_id,
            chunks_used=len(chunks),
            mode=mode,
        )
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=chunks,
            tool_used=None,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent_instance: RAGAgent | None = None


def get_agent() -> RAGAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RAGAgent()
    return _agent_instance