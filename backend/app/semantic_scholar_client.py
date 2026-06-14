from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
import threading
import time

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

from .config import get_settings
from .schemas import Paper


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
USER_AGENT = "PaperReadingAssistant/0.2 (academic-research-assistant; contact: local-app)"
REQUEST_INTERVAL_SECONDS = 1.05
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0
FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "publicationDate",
        "authors",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "openAccessPdf",
        "citationCount",
        "influentialCitationCount",
    ]
)

VENUE_GROUPS: dict[str, list[str]] = {
    "ALL": [],
    "NLP": ["ACL", "EMNLP", "NAACL", "COLING"],
    "ML": ["NeurIPS", "ICML", "ICLR"],
    "AI": ["AAAI", "IJCAI"],
    "CV": ["CVPR", "ICCV", "ECCV"],
    "IR/DM": ["SIGIR", "KDD", "WWW"],
}

VENUE_GROUPS["ALL"] = [
    venue
    for group, venues in VENUE_GROUPS.items()
    if group != "ALL"
    for venue in venues
]

VENUE_ALIASES: dict[str, list[str]] = {
    "NeurIPS": ["NeurIPS", "NIPS", "Neural Information Processing Systems"],
    "ICML": ["ICML", "International Conference on Machine Learning"],
    "ICLR": ["ICLR", "International Conference on Learning Representations"],
    "ACL": ["ACL", "Annual Meeting of the Association for Computational Linguistics"],
    "EMNLP": ["EMNLP", "Empirical Methods in Natural Language Processing"],
    "NAACL": ["NAACL", "North American Chapter of the Association for Computational Linguistics"],
    "COLING": ["COLING", "International Conference on Computational Linguistics"],
    "AAAI": ["AAAI", "AAAI Conference on Artificial Intelligence"],
    "IJCAI": ["IJCAI", "International Joint Conference on Artificial Intelligence"],
    "CVPR": ["CVPR", "Computer Vision and Pattern Recognition"],
    "ICCV": ["ICCV", "International Conference on Computer Vision"],
    "ECCV": ["ECCV", "European Conference on Computer Vision"],
    "SIGIR": ["SIGIR", "Special Interest Group on Information Retrieval"],
    "KDD": ["KDD", "Knowledge Discovery and Data Mining"],
    "WWW": ["WWW", "World Wide Web Conference", "The Web Conference"],
}


def resolve_venues(venue_group: str = "", venues: list[str] | None = None) -> list[str]:
    custom = [venue.strip() for venue in (venues or []) if venue.strip()]
    if custom:
        return custom
    return VENUE_GROUPS.get((venue_group or "").strip(), VENUE_GROUPS["ML"])


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _published_date(paper: dict) -> str:
    if paper.get("publicationDate"):
        return str(paper["publicationDate"])
    if paper.get("year"):
        return f"{paper['year']}-01-01"
    return ""


def _venue_aliases(venues: list[str]) -> list[str]:
    aliases: list[str] = []
    for venue in venues:
        aliases.extend(VENUE_ALIASES.get(venue, [venue]))
    return aliases


def _paper_matches_venue(paper: dict, venues: list[str]) -> bool:
    venue = (paper.get("venue") or "").lower()
    if not venue:
        return False
    return any(item.lower() in venue for item in _venue_aliases(venues))


def _source_tags(paper: dict) -> list[str]:
    tags: list[str] = []
    for item in paper.get("s2FieldsOfStudy") or []:
        if isinstance(item, dict) and item.get("category"):
            tags.append(_clean_text(item.get("category")))
    for item in paper.get("fieldsOfStudy") or []:
        tags.append(_clean_text(str(item)))

    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        key = tag.lower()
        if tag and key not in seen:
            seen.add(key)
            result.append(tag)
        if len(result) >= 5:
            break
    return result


def _paper_to_model(paper: dict) -> Paper:
    external_ids = paper.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv")
    paper_url = paper.get("url") or (f"https://www.semanticscholar.org/paper/{paper.get('paperId')}" if paper.get("paperId") else "")
    pdf_url = ((paper.get("openAccessPdf") or {}).get("url") or "")
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else paper_url
    published = _published_date(paper)
    title = _clean_text(paper.get("title"))
    summary = _clean_text(paper.get("abstract")) or "Semantic Scholar 暂未提供摘要。"
    source_tags = _source_tags(paper)
    return Paper(
        id=paper.get("paperId") or arxiv_id or paper_url,
        title=title,
        authors=[_clean_text(author.get("name")) for author in paper.get("authors", []) if author.get("name")],
        published=published,
        updated=published,
        summary=summary,
        arxiv_url=arxiv_url,
        pdf_url=pdf_url,
        categories=source_tags,
        source="Semantic Scholar",
        venue=_clean_text(paper.get("venue")),
        year=paper.get("year"),
        publication_date=published,
        paper_url=paper_url,
        external_ids=external_ids,
        citation_count=int(paper.get("citationCount") or 0),
        influential_citation_count=int(paper.get("influentialCitationCount") or 0),
        keywords=source_tags,
    )


def _respect_rate_limit() -> None:
    global _last_request_at
    with _rate_limit_lock:
        now = time.monotonic()
        wait_seconds = REQUEST_INTERVAL_SECONDS - (now - _last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _last_request_at = time.monotonic()


def search_semantic_scholar(
    query: str,
    max_results: int,
    days: int,
    venue_group: str = "ML",
    venues: list[str] | None = None,
    sort_by: str = "recommended",
) -> list[Paper]:
    selected_venues = resolve_venues(venue_group, venues)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    year_floor = since.year
    headers = {"User-Agent": USER_AGENT}
    settings = get_settings()
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    collected: list[dict] = []
    last_error: Exception | None = None
    limit = min(max(max_results * 2, 10), 30)

    params = {
        "query": query,
        "fields": FIELDS,
        "year": f"{year_floor}-",
        "venue": ",".join(selected_venues),
    }
    attempt_limits = list(dict.fromkeys([limit, max_results]))
    for attempt_limit in attempt_limits:
        try:
            _respect_rate_limit()
            response = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params={**params, "limit": attempt_limit},
                headers=headers,
                timeout=(6, 25),
            )
            if response.status_code == 429:
                last_error = RuntimeError("Semantic Scholar rate limit exceeded. Configure SEMANTIC_SCHOLAR_API_KEY or try later.")
                break
            response.raise_for_status()
            collected.extend(response.json().get("data", []))
            break
        except Timeout:
            last_error = RuntimeError("Semantic Scholar 请求超时，请稍后重试，或暂时切换到 arXiv 预印本。")
        except (ChunkedEncodingError, ConnectionError):
            last_error = RuntimeError("Semantic Scholar 连接中断，已尝试缩小请求范围；请稍后重试或减少返回数量。")
        except Exception as exc:
            last_error = exc

    if not collected and last_error:
        raise RuntimeError(str(last_error)) from last_error

    deduped: dict[str, dict] = {}
    for paper in collected:
        if not _paper_matches_venue(paper, selected_venues):
            continue
        paper_date = _published_date(paper)
        if paper_date:
            try:
                paper_dt = datetime.fromisoformat(paper_date).replace(tzinfo=timezone.utc)
                if paper_dt < since:
                    continue
            except ValueError:
                pass
        key = paper.get("paperId") or _clean_text(paper.get("title")).lower()
        if key and key not in deduped:
            deduped[key] = paper

    papers = [_paper_to_model(paper) for paper in deduped.values()]
    if sort_by == "citations":
        papers.sort(key=lambda item: (item.citation_count, item.influential_citation_count, item.publication_date), reverse=True)
    else:
        papers.sort(key=lambda item: item.publication_date or str(item.year or ""), reverse=True)
    return papers[:max_results]
