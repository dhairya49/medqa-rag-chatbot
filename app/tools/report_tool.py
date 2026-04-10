"""
app/tools/report_tool.py

Tool 1 — Medical Report Analyser.

Flow:
  1. Extract text from uploaded PDF using PyMuPDF (fitz)
  2. Embed the extracted text and search Qdrant for related medical context
  3. Send report text + retrieved context + user question to LLM
  4. Return a plain-language explanation

Runs in a thread pool executor (called from agent.py via run_in_executor)
so all calls here must be synchronous:
  - embedder.embed_query_sync()  not embed_query()
  - retriever.search_sync()      not search()
  - llm.invoke_sync()            not invoke()

Design notes:
  - Framed strictly as an explanation tool, NOT a diagnostic tool.
  - OCR for scanned PDFs deferred to Phase 4.
  - Max extracted text sent to LLM capped at 3000 chars.
"""

from app.utils.logger import get_logger
from app.models.schemas import SourceChunk

logger = get_logger(__name__)

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

_MAX_REPORT_CHARS = 3000


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from e

    doc        = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            pages_text.append(text.strip())

    doc.close()

    if not pages_text:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned image-based PDF. OCR support coming in Phase 4."
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
    Synchronous — runs inside run_in_executor from agent.py.
    """
    # Step 1: extract text
    try:
        report_text = _extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        logger.warning("pdf_extraction_failed", error=str(exc))
        return {"answer": str(exc), "sources": []}

    # Step 2: truncate if too long
    report_text_truncated = report_text[:_MAX_REPORT_CHARS]
    if len(report_text) > _MAX_REPORT_CHARS:
        logger.info(
            "report_text_truncated",
            original=len(report_text),
            used=_MAX_REPORT_CHARS,
        )

    # Step 3: embed + search — embed_query_sync and search_sync
    # (sync versions because we are already in a thread pool)
    search_query = f"{user_question} {report_text_truncated[:500]}"
    query_vector = embedder.embed_query_sync(search_query)
    chunks: list[SourceChunk] = retriever.search_sync(query_vector, top_k=5)

    # Step 4: build context
    context = (
        "\n\n".join(f"[Source: {c.source}]\n{c.chunk_text}" for c in chunks)
        if chunks else "No additional medical context found."
    )

    # Step 5: invoke_sync — we are in a thread pool, not async context
    prompt = REPORT_PROMPT.format(
        context=context,
        report_text=report_text_truncated,
        question=user_question,
    )
    answer = llm.invoke_sync(prompt)

    logger.info("report_analysis_done", sources_used=len(chunks))
    return {"answer": answer, "sources": chunks}