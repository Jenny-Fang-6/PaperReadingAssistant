import re
from collections import Counter


STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "also",
    "analysis",
    "approach",
    "based",
    "between",
    "datasets",
    "different",
    "efficient",
    "evaluation",
    "experiments",
    "framework",
    "large",
    "learning",
    "model",
    "models",
    "method",
    "methods",
    "paper",
    "performance",
    "propose",
    "proposed",
    "results",
    "show",
    "system",
    "task",
    "tasks",
    "through",
    "using",
    "with",
}


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text or "")
        if word.lower() not in STOPWORDS
    ]
    seen: set[str] = set()
    keywords: list[str] = []
    for word, _ in Counter(words).most_common(limit * 3):
        normalized = word.strip("-")
        if normalized and normalized not in seen:
            seen.add(normalized)
            keywords.append(normalized)
        if len(keywords) >= limit:
            break
    return keywords
