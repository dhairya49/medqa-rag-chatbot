"""
app/tools/report_tool.py

Tool 1 — Medical Report Analyser.

Flow:
  1. Extract text from uploaded PDF using PyMuPDF (fitz)
  2. Build a focused search query from abnormal findings in the report
  3. Embed and search Qdrant for related medical context
  4. Send report text + retrieved context + user question to LLM
  5. Return a plain-language explanation grounded in the actual report values

Runs in a thread pool executor (called from agent.py via run_in_executor)
so all calls here must be synchronous:
  - embedder.embed_query_sync()  not embed_query()
  - retriever.search_sync()      not search()
  - llm.invoke_sync()            not invoke()

Design notes:
  - Report text is the PRIMARY source. RAG context is supplementary.
  - Cover pages and boilerplate are stripped before sending to LLM.
  - Search query is built from abnormal findings, not raw first-500-chars.
  - Max extracted text bumped to 8000 chars to capture full lab tables.
  - OCR for scanned PDFs deferred to Phase 4.
"""

from __future__ import annotations
import re
from app.utils.logger import get_logger
from app.models.schemas import SourceChunk

logger = get_logger(__name__)

# Bumped from 3000 → 8000 to capture full lab value tables in multi-page reports
_MAX_REPORT_CHARS = 8000

# Pages that are pure noise — skip entirely
_SKIP_PAGE_KEYWORDS = [
    "all rights reserved",
    "feedback link",
    "social links",
    "powered by tcpdf",
    "your opinion matters",
    "references",
    "disclaimer",
    "report walkthrough",
    "diet dos and don'ts",
    "fruits and vegetables",
    "nuts and seeds",
    "oils and fats",
]

# Pages containing raw lab tables — highest priority for LLM context
_CLINICAL_TABLE_KEYWORDS = [
    "test name", "result", "unit", "range", "level",
    "clinical data", "your clinical data",
]

# Pages containing abnormal finding summaries — second priority
_FINDING_SUMMARY_KEYWORDS = [
    "need attention", "your result", "cause / effect",
    "health summary", "abnormal", "out of range",
]

# Terms that signal a line contains an actual lab finding worth searching for
_FINDING_KEYWORDS = [
    "abnormal", "out of range", "low", "high", "elevated",
    "deficiency", "borderline", "very abnormal", "deranged",
    "need evaluation", "need attention",
]

REPORT_PROMPT = """\
You are a careful medical assistant explaining a patient's lab report in simple, clear language.

IMPORTANT: The report text below contains the actual test results with values and reference ranges.
Your explanation MUST reference the specific values and ranges found in the report.
Do NOT say "specific values not provided" — they ARE provided in the report text below.

Rules:
- Extract and explain every abnormal or borderline result with its actual value and reference range.
- For normal results, briefly confirm they are within range.
- Do not diagnose the patient or infer conditions not stated in the report.
- If something is unclear or missing from the report, say so directly.
- End by advising the user to review the findings with their doctor.

Required format:

Summary:
- (1-2 sentence overview of overall health picture from the report)

Abnormal / Borderline Findings (include actual value and range for each):
- ...

Normal Findings:
- ...

What To Discuss With A Doctor:
- ...

--- REPORT TEXT (primary source — use these values in your explanation) ---
{report_text}

--- MEDICAL KNOWLEDGE CONTEXT (supplementary — use to explain what findings mean) ---
{context}

--- USER QUESTION ---
{question}

Explanation:"""


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract and prioritize text from PDF pages.

    Priority order sent to LLM:
      1. Clinical data table pages (TEST NAME | RESULT | UNIT | RANGE) — most important
      2. Abnormal finding summary pages (Your Result / Need Attention sections)
      3. Everything else that isn't boilerplate

    This ensures the LLM always receives the actual lab values first, regardless
    of how many pages of explanatory text precede them in the PDF.
    """
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    clinical_pages = []    # priority 1: raw lab tables
    finding_pages = []     # priority 2: abnormal finding summaries
    other_pages = []       # priority 3: everything else useful

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if not text.strip():
            continue

        lower = text.lower()

        # Skip pure boilerplate pages
        if any(kw in lower for kw in _SKIP_PAGE_KEYWORDS):
            continue

        # Classify page by content
        if any(kw in lower for kw in _CLINICAL_TABLE_KEYWORDS):
            clinical_pages.append(text.strip())
        elif any(kw in lower for kw in _FINDING_SUMMARY_KEYWORDS):
            finding_pages.append(text.strip())
        else:
            other_pages.append(text.strip())

    doc.close()

    all_pages = clinical_pages + finding_pages + other_pages

    if not all_pages:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned image-based PDF. OCR support coming in Phase 4."
        )

    full_text = "\n\n".join(all_pages)
    logger.info(
        "pdf_extracted",
        clinical_pages=len(clinical_pages),
        finding_pages=len(finding_pages),
        other_pages=len(other_pages),
        total_chars=len(full_text),
    )
    return full_text


def _build_search_query(user_question: str, report_text: str) -> str:
    """
    Build a focused Qdrant search query from abnormal findings in the report.

    Strategy:
    - Scan report lines for finding keywords (abnormal, low, elevated, etc.)
    - Take the first 10 such lines — these are the clinically relevant terms
    - Combine with the user question for context-aware retrieval

    This avoids the previous bug of using the first 500 chars of the report,
    which was always cover page boilerplate (patient name, logo text, etc.)
    and produced irrelevant Qdrant results.
    """
    lines = report_text.split("\n")
    finding_lines = []

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        lower = line.lower()
        if any(kw in lower for kw in _FINDING_KEYWORDS):
            finding_lines.append(line)
        if len(finding_lines) >= 10:
            break

    # Fallback: if no finding lines detected, use first substantive lines
    if not finding_lines:
        finding_lines = [l.strip() for l in lines if len(l.strip()) > 20][:5]

    findings_str = " ".join(finding_lines)
    query = f"{user_question} {findings_str}"

    # Cap query length to avoid embedding model token overflow
    return query[:500]


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
    # Step 1: Extract text (with boilerplate page filtering)
    try:
        report_text = _extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        logger.warning("pdf_extraction_failed", error=str(exc))
        return {"answer": str(exc), "sources": []}

    # Step 2: Truncate to LLM-safe length (8000 chars captures full lab tables)
    report_text_truncated = report_text[:_MAX_REPORT_CHARS]
    if len(report_text) > _MAX_REPORT_CHARS:
        logger.info(
            "report_text_truncated",
            original=len(report_text),
            used=_MAX_REPORT_CHARS,
        )

    # Step 3: Build a focused search query from actual findings (not cover page noise)
    search_query = _build_search_query(user_question, report_text_truncated)
    logger.info("report_search_query", query_preview=search_query[:120])

    # Step 4: Embed + search Qdrant for supplementary medical context
    query_vector = embedder.embed_query_sync(search_query)
    chunks: list[SourceChunk] = retriever.search_sync(search_query, query_vector, top_k=5)

    # Step 5: Build RAG context (supplementary — report text is primary)
    context = (
        "\n\n".join(f"[Source: {c.source}]\n{c.chunk_text}" for c in chunks)
        if chunks else "No additional medical context retrieved."
    )

    # Step 6: Build prompt — report text comes BEFORE context to establish priority
    prompt = REPORT_PROMPT.format(
        report_text=report_text_truncated,
        context=context,
        question=user_question,
    )

    # Step 7: Call LLM synchronously (we are in a thread pool)
    answer = llm.invoke_sync(prompt)

    logger.info("report_analysis_done", sources_used=len(chunks))
    return {"answer": answer, "sources": chunks}