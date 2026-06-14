from dataclasses import dataclass
from uuid import uuid4

from .embeddings import encode_texts


@dataclass
class StoredPaper:
    paper_id: str
    filename: str
    pages: list[dict]
    chunks: list[dict]
    pdf_bytes: bytes | None = None
    visual_assets: list[dict] | None = None
    visual_asset_images: dict[str, bytes] | None = None


class PaperStore:
    def __init__(self) -> None:
        self._papers: dict[str, StoredPaper] = {}

    def add(self, filename: str, pages: list[dict], chunks: list[dict], pdf_bytes: bytes | None = None) -> StoredPaper:
        paper_id = str(uuid4())
        if chunks:
            embeddings = encode_texts([chunk["text"] for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
        paper = StoredPaper(
            paper_id=paper_id,
            filename=filename,
            pages=pages,
            chunks=chunks,
            pdf_bytes=pdf_bytes,
            visual_assets=None,
            visual_asset_images=None,
        )
        self._papers[paper_id] = paper
        return paper

    def get(self, paper_id: str) -> StoredPaper:
        if paper_id not in self._papers:
            raise KeyError(f"Paper {paper_id} not found. Upload the PDF again.")
        return self._papers[paper_id]

    def set_visual_assets(self, paper_id: str, assets: list[dict], images: dict[str, bytes]) -> None:
        paper = self.get(paper_id)
        paper.visual_assets = assets
        paper.visual_asset_images = images


paper_store = PaperStore()
