"""
app/ingestion/drug_loader.py

Build an offline drug corpus from drugs.com and openFDA for the drug tool.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)

_DRUGS_COM_BASE = "https://www.drugs.com"
_OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.drugs.com/",
}
_SECTION_KEYWORDS = {
    "uses": ["uses", "what is", "what is this medicine"],
    "dosage": ["dosage", "how should i take", "usual adult dose"],
    "warnings": ["warnings", "before taking", "precautions", "contraindications"],
    "side_effects": ["side effects", "adverse reactions"],
    "interactions": ["interactions", "drug interactions"],
}
_OPENFDA_FIELDS = {
    "uses": "indications_and_usage",
    "dosage": "dosage_and_administration",
    "warnings": "warnings",
    "side_effects": "adverse_reactions",
    "interactions": "drug_interactions",
}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _split_text(text: str, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind(". ", start, end)
            if split_at > start + 200:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_drug_name(title: str) -> str:
    title = _clean_text(title)
    title = re.sub(
        r"\s+(uses|dosage|side effects|warnings|interactions).*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return title.strip(" -")


def _collect_section_text(heading) -> str:
    parts: list[str] = []
    for sibling in heading.find_next_siblings():
        if sibling.name in {"h1", "h2", "h3"}:
            break
        text = sibling.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return _clean_text(" ".join(parts))


def _extract_drugs_com_sections(soup: BeautifulSoup) -> dict[str, str]:
    sections: dict[str, str] = {}

    for heading in soup.find_all(["h2", "h3"]):
        heading_text = _clean_text(heading.get_text(" ", strip=True)).lower()
        section_text = _collect_section_text(heading)
        if not section_text:
            continue

        for section_name, keywords in _SECTION_KEYWORDS.items():
            if section_name in sections:
                continue
            if any(keyword in heading_text for keyword in keywords):
                sections[section_name] = section_text
                break

    if not sections:
        paragraphs = [
            _clean_text(p.get_text(" ", strip=True))
            for p in soup.find_all("p")
            if len(_clean_text(p.get_text(" ", strip=True))) > 80
        ]
        if paragraphs:
            sections["overview"] = " ".join(paragraphs[:8])

    return sections


def _get_alpha_links(client: httpx.Client, limit: int | None) -> list[str]:
    links: list[str] = []
    settings = get_settings()

    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"{_DRUGS_COM_BASE}/alpha/{letter}.html"
        response = client.get(url, headers=_HEADERS, timeout=settings.drug_request_timeout)
        if response.status_code == 403:
            raise RuntimeError(
                "drugs.com blocked the ingestion request with HTTP 403. "
                "The site is rejecting the current scraper fingerprint/IP."
            )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.select("ul.ddc-list-column-2 li a"):
            href = anchor.get("href")
            if href and href.startswith("/"):
                links.append(f"{_DRUGS_COM_BASE}{href}")

        if limit and len(set(links)) >= limit:
            break

        time.sleep(settings.drug_scrape_delay_seconds)

    unique_links = sorted(set(links))
    return unique_links[:limit] if limit else unique_links


def _scrape_drugs_com_record(client: httpx.Client, url: str) -> dict[str, Any] | None:
    settings = get_settings()
    response = client.get(url, headers=_HEADERS, timeout=settings.drug_request_timeout)
    if response.status_code == 403:
        log.warning("drug_page_blocked", url=url, status=response.status_code)
        return None
    if response.status_code != 200:
        log.warning("drug_page_skipped", url=url, status=response.status_code)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1")
    if not title:
        return None

    drug_name = _extract_drug_name(title.get_text(" ", strip=True))
    sections = _extract_drugs_com_sections(soup)
    if not sections:
        return None

    return {
        "drug_name": drug_name,
        "lookup_names": {_normalise_name(drug_name)},
        "source_url": url,
        "drugs_com_sections": sections,
    }


def _fetch_openfda_record(client: httpx.Client, drug_name: str) -> dict[str, Any] | None:
    settings = get_settings()
    search_variants = [
        ("openfda.generic_name", drug_name),
        ("openfda.brand_name", drug_name),
        ("openfda.substance_name", drug_name),
    ]

    for field, value in search_variants:
        response = client.get(
            _OPENFDA_LABEL_URL,
            params={"search": f'{field}:"{value}"', "limit": 1},
            headers=_HEADERS,
            timeout=settings.drug_request_timeout,
        )
        if response.status_code != 200:
            continue

        data = response.json()
        results = data.get("results", [])
        if not results:
            continue

        label = results[0]
        openfda = label.get("openfda", {})
        sections: dict[str, str] = {}
        for section_name, field_name in _OPENFDA_FIELDS.items():
            value_list = label.get(field_name)
            if value_list and isinstance(value_list, list):
                sections[section_name] = _clean_text(value_list[0])

        if not sections:
            return None

        aliases = set()
        for key in ("generic_name", "brand_name", "substance_name"):
            for item in openfda.get(key, []):
                cleaned = _normalise_name(item)
                if cleaned:
                    aliases.add(cleaned)

        source_url = (
            f"{_OPENFDA_LABEL_URL}?search="
            f'{field}:"{value}"&limit=1'
        )

        return {
            "lookup_names": aliases,
            "source_url": source_url,
            "openfda_sections": sections,
        }

    return None


def _make_source_chunks(
    *,
    drug_name: str,
    lookup_names: set[str],
    sections: dict[str, str],
    source: str,
    source_url: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for section_name, section_text in sections.items():
        parts = _split_text(section_text)
        for idx, part in enumerate(parts):
            chunks.append(
                {
                    "text": (
                        f"Drug: {drug_name}\n"
                        f"Section: {section_name.replace('_', ' ')}\n"
                        f"Source: {source}\n"
                        f"Content: {part}"
                    ),
                    "question": f"{drug_name} {section_name}",
                    "source": source,
                    "category": "drug",
                    "topic": drug_name,
                    "chunk_index": idx,
                    "total_chunks": len(parts),
                    "token_count": len(part.split()),
                    "drug_name": drug_name,
                    "drug_name_normalized": _normalise_name(drug_name),
                    "lookup_names": sorted(lookup_names),
                    "section": section_name,
                    "source_url": source_url,
                    "source_type": source,
                }
            )

    return chunks


def load_drug_chunks(limit: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    effective_limit = limit if limit is not None else settings.drug_ingestion_limit or None

    log.info("drug_ingestion_start", limit=effective_limit or "all")
    chunks: list[dict[str, Any]] = []

    with httpx.Client(
        follow_redirects=True,
        headers=_HEADERS,
        http2=False,
    ) as client:
        links = _get_alpha_links(client, effective_limit)
        log.info("drug_links_loaded", count=len(links))

        for index, url in enumerate(links, start=1):
            try:
                scraped = _scrape_drugs_com_record(client, url)
                if not scraped:
                    continue

                openfda = _fetch_openfda_record(client, scraped["drug_name"])
                lookup_names = set(scraped["lookup_names"])
                if openfda:
                    lookup_names.update(openfda["lookup_names"])

                chunks.extend(
                    _make_source_chunks(
                        drug_name=scraped["drug_name"],
                        lookup_names=lookup_names,
                        sections=scraped["drugs_com_sections"],
                        source="drugs.com",
                        source_url=scraped["source_url"],
                    )
                )

                if openfda:
                    chunks.extend(
                        _make_source_chunks(
                            drug_name=scraped["drug_name"],
                            lookup_names=lookup_names,
                            sections=openfda["openfda_sections"],
                            source="openFDA",
                            source_url=openfda["source_url"],
                        )
                    )

                if index % 25 == 0:
                    log.info("drug_ingestion_progress", processed=index, chunks=len(chunks))

                time.sleep(settings.drug_scrape_delay_seconds)

            except Exception as exc:
                log.warning("drug_ingestion_item_failed", url=url, error=str(exc))

    log.info("drug_ingestion_loaded", total_chunks=len(chunks))
    return chunks
