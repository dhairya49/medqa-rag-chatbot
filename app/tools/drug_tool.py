"""
app/tools/drug_tool.py

Tool 2 — Drug Information Retriever.

Flow:
  1. Scrape drugs.com for the given drug name (primary source)
  2. If drugs.com fails or returns no useful content → fall back to FDA API
  3. Send scraped info + user question to Llama 3.1 8B
  4. Return answer with source URL cited

Design notes:
  - Uses httpx for async-compatible HTTP requests (sync mode here, async in Phase 4)
  - Uses BeautifulSoup for HTML parsing
  - FDA fallback uses the official openFDA REST API (no scraping, JSON response)
  - LLM is given the raw scraped text as context — it summarises and answers
  - Source URL is always included in the response for transparency
  - If both sources fail, returns a clear failure message without hallucinating

Depends on:
  - httpx               — HTTP client
  - beautifulsoup4      — HTML parsing for drugs.com
  - app/services/llm.py — LLM generation
"""

import httpx
from bs4 import BeautifulSoup

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DRUGS_COM_BASE = "https://www.drugs.com"
_FDA_API_BASE = "https://api.fda.gov/drug/label.json"
_REQUEST_TIMEOUT = 10  # seconds
_MAX_SCRAPED_CHARS = 3000

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MedQA-RAG-Bot/1.0; "
        "educational research purposes)"
    )
}

# ── Prompt template ───────────────────────────────────────────────────────────

DRUG_PROMPT = """\
You are a helpful medical assistant providing factual drug information \
from verified sources. Explain clearly in plain language suitable for a \
general audience.

Important:
- Only use the information provided below. Do not add information not present.
- Always remind the user to consult their doctor or pharmacist before \
taking any medication.
- Do not recommend dosages — only describe what the source states.

Drug information from {source_url}:
{drug_info}

User question:
{question}

Answer:"""


# ── drugs.com scraper ─────────────────────────────────────────────────────────

def _scrape_drugs_com(drug_name: str) -> tuple[str | None, str | None]:
    """
    Scrape drugs.com for drug information.

    Returns:
        (text_content, source_url) if successful
        (None, None) if failed
    """
    # drugs.com uses lowercase hyphenated URLs e.g. /metformin.html
    slug = drug_name.lower().replace(" ", "-")
    url = f"{_DRUGS_COM_BASE}/{slug}.html"

    try:
        response = httpx.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT, follow_redirects=True)

        if response.status_code != 200:
            logger.warning("drugs_com_not_found", drug=drug_name, status=response.status_code)
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")

        # drugs.com stores the main drug info in .drugHeader and .contentBox divs
        content_sections = []

        # Drug header (name, generic name, drug class)
        header = soup.find("div", class_="drugHeader")
        if header:
            content_sections.append(header.get_text(separator=" ", strip=True))

        # Main content sections (uses, warnings, side effects, dosage)
        for section in soup.find_all("div", class_=["contentBox", "drug-content"]):
            text = section.get_text(separator=" ", strip=True)
            if len(text) > 50:  # skip empty/tiny sections
                content_sections.append(text)

        if not content_sections:
            # fallback: grab all paragraph text
            paragraphs = soup.find_all("p")
            content_sections = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40]

        if not content_sections:
            logger.warning("drugs_com_no_content", drug=drug_name)
            return None, None

        full_text = "\n\n".join(content_sections)[:_MAX_SCRAPED_CHARS]
        logger.info("drugs_com_scraped", drug=drug_name, chars=len(full_text))
        return full_text, url

    except httpx.RequestError as exc:
        logger.warning("drugs_com_request_error", drug=drug_name, error=str(exc))
        return None, None


# ── FDA API fallback ──────────────────────────────────────────────────────────

def _fetch_fda(drug_name: str) -> tuple[str | None, str | None]:
    """
    Query the openFDA drug label API as fallback.
    Returns (text_content, source_url) or (None, None).
    """
    params = {
        "search": f'openfda.brand_name:"{drug_name}"',
        "limit": 1,
    }
    source_url = f"{_FDA_API_BASE}?search=openfda.brand_name:{drug_name}&limit=1"

    try:
        response = httpx.get(
            _FDA_API_BASE,
            params=params,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            logger.warning("fda_api_not_found", drug=drug_name, status=response.status_code)
            return None, None

        data = response.json()
        results = data.get("results", [])

        if not results:
            logger.warning("fda_api_empty_results", drug=drug_name)
            return None, None

        label = results[0]

        # Extract the most useful fields from the FDA label
        sections = []
        for field in [
            "indications_and_usage",
            "warnings",
            "adverse_reactions",
            "dosage_and_administration",
            "contraindications",
            "drug_interactions",
        ]:
            value = label.get(field)
            if value and isinstance(value, list):
                sections.append(f"{field.replace('_', ' ').title()}:\n{value[0][:500]}")

        if not sections:
            return None, None

        full_text = "\n\n".join(sections)[:_MAX_SCRAPED_CHARS]
        logger.info("fda_api_fetched", drug=drug_name, chars=len(full_text))
        return full_text, source_url

    except (httpx.RequestError, KeyError, ValueError) as exc:
        logger.warning("fda_api_error", drug=drug_name, error=str(exc))
        return None, None


# ── Main entry point ──────────────────────────────────────────────────────────

def lookup_drug(
    drug_name: str,
    user_question: str,
    llm,
) -> dict:
    """
    Full drug lookup pipeline with fallback.

    Args:
        drug_name    : detected drug name from user message
        user_question: full user message for context
        llm          : LLMService instance (from agent)

    Returns:
        dict with keys: answer (str), source_url (str)
    """

    # Step 1: try drugs.com first
    drug_info, source_url = _scrape_drugs_com(drug_name)

    # Step 2: fall back to FDA if drugs.com failed
    if not drug_info:
        logger.info("falling_back_to_fda", drug=drug_name)
        drug_info, source_url = _fetch_fda(drug_name)

    # Step 3: if both failed, return honest failure message
    if not drug_info:
        logger.warning("all_drug_sources_failed", drug=drug_name)
        return {
            "answer": (
                f"I was unable to retrieve information about '{drug_name}' "
                f"from drugs.com or the FDA database at this time. "
                f"Please visit https://www.drugs.com or https://www.fda.gov "
                f"directly, or consult your pharmacist."
            ),
            "source_url": None,
        }

    # Step 4: build prompt and call LLM
    prompt = DRUG_PROMPT.format(
        source_url=source_url,
        drug_info=drug_info,
        question=user_question,
    )

    answer = llm.invoke(prompt)
    logger.info("drug_lookup_done", drug=drug_name, source=source_url)

    return {
        "answer": answer,
        "source_url": source_url,
    }