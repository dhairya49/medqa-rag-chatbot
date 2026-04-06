"""
app/tools/report_tool.py

Tool 1 — Medical Report Analyser.

Flow:
  1. Extract text from uploaded PDF using PyMuPDF (fitz)
  2. Embed the extracted text and search Qdrant for related medical context
  3. Send report text + retrieved context + user question to Llama 3.1 8B
  4. Return a plain-language explanation

Design notes:
  - Framed strictly as an explanation tool, NOT a diagnostic tool.
    The prompt explicitly tells the LLM not to diagnose.
  - OCR for scanned PDFs is deferred to Phase 4.
  - Max extracted text sent to LLM is capped at 3000 chars to stay within
    Llama 3.1 8B context window comfortably alongside the retrieved context.

Depends on:
  - PyMuPDF (fitz)          — PDF text extraction
  - app/services/retrieval.py  — Qdrant search
  - app/services/embedding.py  — embed report text for retrieval
  - app/services/llm.py        — LLM generation
  - app/models/schemas.py      — SourceChunk
"""

from app.utils.logger import get_logger
from app.models.schemas import SourceChunk

logger = get_logger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

REPORT_PROMPT = """\
You are a helpful medical assistant explaining a patient's medical report \
in simple, clear language that anyone can understand.

You are NOT diagnosing the patient. You are only explaining what the report says.
Always remind the user to discuss the results with their doctor.

Medical knowledge context (from verified sources):
{context}

Extracted report content:
{report_text}

User's question about the report:
{question}

Explanation:"""

# Maximum characters of report text to send to the LLM
_MAX_REPORT_CHARS = 3000


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    Returns concatenated text from all pages.
    Raises ValueError if the PDF has no extractable text (likely scanned).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF is required for report analysis. Install with: pip install pymupdf") from e

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text.strip())

    doc.close()

    if not pages_text:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned image-based PDF. OCR support is coming in Phase 4."
        )

    full_text = "\n\n".join(pages_text)
    logger.info("pdf_extracted", pages=len(pages_text), total_chars=len(full_text))
    return full_text


def analyse_report(
    pdf_bytes: bytes,
    user_question: str,
    retriever,
    embedder,
    llm,
) -> dict:
    """
    Full report analysis pipeline.

    Args:
        pdf_bytes     : raw bytes of the uploaded PDF
        user_question : user's question about the report
        retriever     : RetrievalService instance (from agent)
        embedder      : EmbeddingService instance (from agent)
        llm           : LLMService instance (from agent)

    Returns:
        dict with keys: answer (str), sources (list[SourceChunk])
    """

    # Step 1: extract text from PDF
    try:
        report_text = _extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        logger.warning("pdf_extraction_failed", error=str(exc))
        return {
            "answer": str(exc),
            "sources": [],
        }

    # Step 2: truncate report text if too long
    report_text_truncated = report_text[:_MAX_REPORT_CHARS]
    if len(report_text) > _MAX_REPORT_CHARS:
        logger.info("report_text_truncated", original=len(report_text), used=_MAX_REPORT_CHARS)

    # Step 3: embed the report text and search for related medical context
    # We embed a combination of the question + first 500 chars of report
    # so retrieval is guided by both what the user asked and what the report says
    search_query = f"{user_question} {report_text_truncated[:500]}"
    query_vector = embedder.embed_query(search_query)
    chunks: list[SourceChunk] = retriever.search(query_vector, top_k=5)

    # Step 4: build context from retrieved chunks
    context = "\n\n".join(
        f"[Source: {c.source}]\n{c.chunk_text}" for c in chunks
    ) if chunks else "No additional medical context found."

    # Step 5: build and send prompt to LLM
    prompt = REPORT_PROMPT.format(
        context=context,
        report_text=report_text_truncated,
        question=user_question,
    )

    answer = llm.invoke(prompt)

    logger.info("report_analysis_done", sources_used=len(chunks))

    return {
        "answer": answer,
        "sources": chunks,
    }