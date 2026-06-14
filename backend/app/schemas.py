from typing import Any, Optional

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    name: str
    status: str = Field(pattern="^(pending|running|completed|failed)$")
    detail: str = ""
    elapsed_ms: int = 0


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    published: str
    updated: Optional[str] = None
    summary: str
    arxiv_url: str
    pdf_url: str
    categories: list[str] = []
    source: str = "arXiv"
    venue: str = ""
    year: Optional[int] = None
    publication_date: str = ""
    paper_url: str = ""
    external_ids: dict[str, Any] = {}
    citation_count: int = 0
    influential_citation_count: int = 0
    keywords: list[str] = []


class RankedPaper(Paper):
    score: float
    relevance_score: float
    recency_score: float
    keyword_score: float
    code_score: float
    citation_score: float = 0.0
    explanation: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)
    days: int = Field(default=365, ge=1, le=3650)


class SearchResponse(BaseModel):
    papers: list[Paper]
    trace: list[TraceStep]


class RankRequest(BaseModel):
    query: str = Field(min_length=1)
    papers: list[Paper]


class RankResponse(BaseModel):
    papers: list[RankedPaper]
    trace: list[TraceStep]


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    page_count: int
    chunk_count: int
    char_count: int
    trace: list[TraceStep]


class PdfUrlRequest(BaseModel):
    pdf_url: str = Field(min_length=1)
    title: str = "arxiv_paper"


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    stage: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    content: str = ""
    trace: list[TraceStep] = []


class EvidenceChunk(BaseModel):
    page: int
    text: str
    score: float


class VisualPage(BaseModel):
    page: int
    label: str = ""
    reason: str = ""


class VisualAsset(BaseModel):
    asset_id: str
    asset_type: str
    page: int
    label: str = ""
    caption: str = ""
    reason: str = ""
    source: str = "pymupdf_fallback"
    confidence: str = "low"
    bbox: list[float] = []
    selection_source: str = ""
    vision_reason: str = ""
    candidate_count: int = 0


class AnalysisResponse(BaseModel):
    content: str
    visual_assets: list[VisualAsset] = []
    trace: list[TraceStep]


class VisualAssetsResponse(BaseModel):
    assets: list[VisualAsset]
    trace: list[TraceStep]


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=8)


class QAResponse(BaseModel):
    answer: str
    evidence: list[EvidenceChunk]
    visual_pages: list[VisualPage] = []
    trace: list[TraceStep]


class TrendsRequest(BaseModel):
    papers: list[Paper]
    query: str = ""


class TrendTopic(BaseModel):
    name: str
    paper_count: int
    paper_ids: list[str]


class TrendsResponse(BaseModel):
    topics: list[TrendTopic]
    time_distribution: dict[str, int]
    llm_analysis: str
    trace: list[TraceStep]


class GraphRequest(BaseModel):
    papers: list[Paper]
    similarity_threshold: float = Field(default=0.38, ge=0.0, le=1.0)


class GraphNode(BaseModel):
    id: str
    title: str
    category: str = ""
    published: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float
    reason: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    trace: list[TraceStep]


class LiteratureReviewRequest(BaseModel):
    papers: list[Paper]
    graph: Optional[dict[str, Any]] = None
    query: str = ""


class WatchTopicRequest(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)
    days: int = Field(default=30, ge=1, le=3650)
    seen_ids: list[str] = []
    source_mode: str = Field(default="conference", pattern="^(conference|preprint|hybrid)$")
    venue_group: str = "ML"
    venues: list[str] = []
    sort_by: str = Field(default="recommended", pattern="^(latest|citations|recommended)$")
    baseline_done: bool = False


class WatchRefreshRequest(BaseModel):
    topics: list[WatchTopicRequest]


class WatchPaper(RankedPaper):
    is_new: bool = False


class WatchTopicResult(BaseModel):
    id: str
    name: str
    query: str
    source_mode: str = "conference"
    venue_group: str = ""
    venues: list[str] = []
    checked_at: str
    papers: list[WatchPaper]
    new_count: int
    total_count: int
    baseline_count: int = 0
    status_message: str = ""
    source_warning: str = ""
    sort_by: str = "recommended"
    sort_warning: str = ""
    error: Optional[str] = None


class WatchRefreshResponse(BaseModel):
    topics: list[WatchTopicResult]
    trace: list[TraceStep]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_model_loaded: bool
    llm_configured: bool
    semantic_scholar_configured: bool
