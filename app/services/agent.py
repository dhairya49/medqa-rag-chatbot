"""
app/services/agent.py

Async RAG agent — all blocking calls run in thread pools.
Handles 10+ concurrent requests without blocking the event loop.

Two response modes:
  concise  — 150 token hard limit, fast (~8-10s locally)
  detailed — full response, slower (~40-120s locally)
"""

import asyncio
import re
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.services.embedding import get_embedding_service
from app.services.retrieval import get_retrieval_service
from app.services.llm import get_llm_service
from app.models.schemas import ChatResponse, SourceChunk

logger = get_logger(__name__)

# ── Token limits per mode ─────────────────────────────────────────────────────

_CONCISE_TOKENS = 150

# ── Prompts ───────────────────────────────────────────────────────────────────

RAG_PROMPT_CONCISE = """\
You are a helpful medical assistant. Answer the user's question using only \
the medical information provided in the context below.

Guidelines:
- Answer in 3-5 sentences maximum. Be brief and clear.
- Use simple, plain language suitable for a general audience.
- If the context does not contain enough information, say so in one sentence.
- Do not make up information. Do not go beyond what the context states.
- If the question involves symptoms or diagnosis, remind the user to consult a doctor.

Context:
{context}

Question:
{question}

Answer:"""

RAG_PROMPT_DETAILED = """\
You are a helpful medical assistant. Answer the user's question clearly \
and accurately using only the medical information provided in the context below.

Guidelines:
- Explain in simple, plain language suitable for a general audience.
- If the context does not contain enough information, say so honestly.
- Do not make up information. Do not go beyond what the context states.
- If the question involves symptoms or diagnosis, remind the user to consult a doctor.

Context:
{context}

Question:
{question}

Answer:"""

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
        self._settings = get_settings()
        self._embedder = get_embedding_service()
        self._retriever = get_retrieval_service()
        self._llm = get_llm_service()
        logger.info("rag_agent_initialised")

    async def run(
        self,
        session_id: str,
        message: str,
        top_k: int | None = None,
        pdf_bytes: bytes | None = None,
        mode: str = "concise",
    ) -> ChatResponse:
        """
        Main entry point for every /chat request.

        mode: "concise"  → short answer, fast
              "detailed" → full answer, slower
        """
        # ── Tool 1: PDF report ────────────────────────────────────────────────
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

        # ── Tool 2: Drug lookup ───────────────────────────────────────────────
        drug_name = _detect_drug_name(message)
        if drug_name:
            logger.info("routing_to_drug_tool", session_id=session_id, drug=drug_name)
            from app.tools.drug_tool import lookup_drug
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: lookup_drug(
                    drug_name=drug_name,
                    user_question=message,
                    llm=self._llm,
                )
            )
            return ChatResponse(
                session_id=session_id,
                answer=result["answer"],
                sources=[],
                tool_used="drug_tool",
            )

        # ── RAG path ──────────────────────────────────────────────────────────
        logger.info("routing_to_rag", session_id=session_id, mode=mode)
        return await self._rag_answer(session_id, message, top_k, mode)

    async def _rag_answer(
        self,
        session_id: str,
        message: str,
        top_k: int | None,
        mode: str,
    ) -> ChatResponse:
        # Step 1: embed query
        query_vector = await self._embedder.embed_query(message)

        # Step 2: retrieve chunks
        chunks: list[SourceChunk] = await self._retriever.search(
            query_vector, top_k=top_k
        )

        if not chunks:
            logger.warning("no_chunks_retrieved", session_id=session_id)
            return ChatResponse(
                session_id=session_id,
                answer="I could not find relevant medical information for your question. Please consult a healthcare professional.",
                sources=[],
                tool_used=None,
            )

        # Step 3: build context
        context = "\n\n".join(
            f"[Source: {c.source}]\n{c.chunk_text}" for c in chunks
        )

        # Step 4: select prompt and token limit based on mode
        if mode == "concise":
            prompt = RAG_PROMPT_CONCISE.format(context=context, question=message)
            max_tokens = _CONCISE_TOKENS
        else:
            prompt = RAG_PROMPT_DETAILED.format(context=context, question=message)
            max_tokens = self._settings.llm_max_tokens

        # Step 5: call LLM
        answer = await self._llm.invoke(prompt, max_tokens=max_tokens)

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