import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from http.client import HTTPSConnection
from urllib.parse import quote_plus

import requests

from .schemas import Paper


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HOST = "export.arxiv.org"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "PaperReadingAssistant/0.1 (academic-research-assistant; contact: local-app)"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _paper_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _fetch_arxiv(params: str) -> str:
    url = f"{ARXIV_API_URL}?{params}"
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=25,
                headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml"},
            )
            if response.status_code == 429:
                time.sleep(3 + attempt * 4)
                continue
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt * 2)

    # Docker Desktop networks sometimes make requests report a closed connection
    # while arXiv is still returning a useful HTTP status. Use stdlib as fallback.
    try:
        connection = HTTPSConnection(ARXIV_HOST, 443, timeout=25)
        connection.request(
            "GET",
            f"/api/query?{params}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml", "Connection": "close"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        connection.close()
    except Exception as exc:
        raise RuntimeError("arXiv is temporarily unavailable or rate-limited. Please wait and retry.") from exc
    if response.status == 429:
        raise RuntimeError("arXiv rate limit exceeded. Please wait a minute and retry.")
    if response.status >= 400:
        raise RuntimeError(f"arXiv request failed with HTTP {response.status}: {body[:200]}")
    if not body and last_error:
        raise last_error
    return body


def search_arxiv(query: str, max_results: int, days: int) -> list[Paper]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    params = (
        f"search_query=all:{quote_plus(query)}"
        f"&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    root = ET.fromstring(_fetch_arxiv(params))

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        published = _clean_text(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        if published:
            try:
                published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if published_dt < since:
                    continue
            except ValueError:
                pass

        arxiv_url = _clean_text(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_url:
            pdf_url = arxiv_url.replace("/abs/", "/pdf/")

        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        categories = [category.attrib.get("term", "") for category in entry.findall("atom:category", ATOM_NS)]
        papers.append(
            Paper(
                id=_paper_id_from_url(arxiv_url),
                title=title,
                authors=[
                    _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
                    for author in entry.findall("atom:author", ATOM_NS)
                ],
                published=published,
                updated=_clean_text(entry.findtext("atom:updated", default="", namespaces=ATOM_NS)),
                summary=summary,
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
                categories=categories,
                source="arXiv",
                venue="arXiv",
                year=int(published[:4]) if published[:4].isdigit() else None,
                publication_date=published[:10],
                paper_url=arxiv_url,
                external_ids={"ArXiv": _paper_id_from_url(arxiv_url)},
                keywords=categories[:5],
            )
        )
    return papers
