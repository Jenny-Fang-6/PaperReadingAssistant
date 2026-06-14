from io import BytesIO
import time
from typing import BinaryIO

import requests
from pypdf import PdfReader

from .config import get_settings


def parse_pdf_bytes(raw: bytes) -> tuple[list[dict], int, int]:
    reader = PdfReader(BytesIO(raw))
    pages: list[dict] = []
    char_count = 0
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if text:
            pages.append({"page": index, "text": text})
            char_count += len(text)
    return pages, len(reader.pages), char_count


def parse_pdf(file_obj: BinaryIO) -> tuple[list[dict], int, int]:
    return parse_pdf_bytes(file_obj.read())


def download_pdf(pdf_url: str) -> BytesIO:
    urls = [pdf_url]
    if "arxiv.org/pdf/" in pdf_url and "export.arxiv.org" not in pdf_url:
        urls.append(pdf_url.replace("https://arxiv.org/pdf/", "https://export.arxiv.org/pdf/"))

    last_error: Exception | None = None
    for url in urls:
        for attempt in range(4):
            try:
                response = requests.get(
                    url,
                    timeout=60,
                    headers={
                        "User-Agent": "PaperReadingAssistant/0.1 (academic-research-assistant)",
                        "Accept": "application/pdf,*/*",
                    },
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
                    raise ValueError(f"The URL did not return a PDF document: {content_type or 'unknown content type'}")
                return BytesIO(response.content)
            except Exception as exc:
                last_error = exc
                time.sleep(1 + attempt * 2)
    raise RuntimeError(f"Failed to download PDF after retries: {last_error}")


def chunk_pages(pages: list[dict], chunk_size: int = 1200, overlap: int = 180) -> list[dict]:
    settings = get_settings()
    chunks: list[dict] = []
    seen_chars = 0
    for page in pages:
        text = page["text"][: max(settings.max_pdf_chars - seen_chars, 0)]
        seen_chars += len(text)
        if not text:
            break
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"page": page["page"], "text": chunk_text})
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
        if seen_chars >= settings.max_pdf_chars:
            break
    return chunks
