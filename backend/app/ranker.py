import math
import re
from datetime import datetime, timezone

import numpy as np

from .embeddings import encode_texts
from .schemas import Paper, RankedPaper


CODE_TERMS = ("github", "code", "repository", "implementation", "open-source", "open source")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)}


def _recency_score(published: str) -> float:
    try:
        published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return 0.3
    if published_dt.tzinfo is None:
        published_dt = published_dt.replace(tzinfo=timezone.utc)
    days = max((datetime.now(timezone.utc) - published_dt).days, 0)
    return float(math.exp(-days / 365))


def rank_papers(query: str, papers: list[Paper]) -> list[RankedPaper]:
    if not papers:
        return []

    texts = [f"{paper.title}. {paper.summary}" for paper in papers]
    vectors = encode_texts([query] + texts)
    query_vector = vectors[0]
    paper_vectors = vectors[1:]
    similarities = np.matmul(paper_vectors, query_vector)
    min_sim, max_sim = float(similarities.min()), float(similarities.max())
    denom = max(max_sim - min_sim, 1e-6)
    normalized_sims = (similarities - min_sim) / denom

    query_tokens = _tokens(query)
    citations = np.array([paper.citation_count for paper in papers], dtype=float)
    citation_denom = max(float(citations.max() - citations.min()), 1e-6)
    citation_scores = (citations - float(citations.min())) / citation_denom
    ranked: list[RankedPaper] = []
    for paper, relevance, citation_score in zip(papers, normalized_sims, citation_scores):
        paper_text = f"{paper.title} {paper.summary}".lower()
        paper_tokens = _tokens(paper_text)
        keyword_score = len(query_tokens & paper_tokens) / max(len(query_tokens), 1)
        code_score = 1.0 if any(term in paper_text for term in CODE_TERMS) else 0.0
        recency = _recency_score(paper.published)
        score = 0.42 * float(relevance) + 0.22 * recency + 0.14 * keyword_score + 0.1 * code_score + 0.12 * float(citation_score)
        explanation = (
            f"相关性 {float(relevance):.2f}，新近性 {recency:.2f}，"
            f"关键词匹配 {keyword_score:.2f}，代码线索 {code_score:.2f}，"
            f"引用量 {paper.citation_count}"
        )
        ranked.append(
            RankedPaper(
                **paper.model_dump(),
                score=round(score, 4),
                relevance_score=round(float(relevance), 4),
                recency_score=round(recency, 4),
                keyword_score=round(keyword_score, 4),
                code_score=round(code_score, 4),
                citation_score=round(float(citation_score), 4),
                explanation=explanation,
            )
        )

    return sorted(ranked, key=lambda item: item.score, reverse=True)
