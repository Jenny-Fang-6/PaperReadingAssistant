from datetime import datetime, timezone
from io import BytesIO
import json
import time

import fitz
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .agent_trace import AgentTrace
from .analysis import (
    answer_question,
    build_graph,
    build_topics,
    explain_trends,
    filter_evidence_chunks,
    literature_review,
    reviewer_analysis,
    reviewer_prompt,
    summarize_pdf,
    summary_prompt,
    time_distribution,
)
from .arxiv_client import search_arxiv
from .config import get_settings
from .embeddings import get_embedding_model, model_loaded, top_k
from .job_store import job_store
from .llm_client import LLMConfigurationError, llm_configured, stream_complete
from .paper_store import paper_store
from .pdf_parser import chunk_pages, parse_pdf, parse_pdf_bytes
from .pdf_parser import download_pdf
from .ranker import rank_papers
from .semantic_scholar_client import resolve_venues, search_semantic_scholar
from .schemas import (
    AnalysisResponse,
    GraphRequest,
    GraphResponse,
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    LiteratureReviewRequest,
    PdfUrlRequest,
    QARequest,
    QAResponse,
    RankRequest,
    RankResponse,
    SearchRequest,
    SearchResponse,
    TrendsRequest,
    TrendsResponse,
    UploadResponse,
    VisualAssetsResponse,
    WatchPaper,
    WatchRefreshRequest,
    WatchRefreshResponse,
    WatchTopicResult,
)
from .visual_assets import extract_visual_assets, visual_assets_prompt


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_embedding_model() -> None:
    get_embedding_model()


def _http_error(exc: Exception) -> HTTPException:
    status = 400 if isinstance(exc, (KeyError, ValueError, LLMConfigurationError)) else 502
    return HTTPException(status_code=status, detail=str(exc))


def _title_key(title: str) -> str:
    return " ".join(title.lower().split())


def _mark_arxiv_supplement(papers: list) -> list:
    for paper in papers:
        paper.source = "arXiv 补充"
        paper.venue = paper.venue or "arXiv 补充"
    return papers


def _topic_venues(topic) -> list[str]:
    if topic.source_mode == "preprint":
        return []
    return resolve_venues(topic.venue_group, topic.venues)


def _semantic_scholar_reason(exc: Exception) -> str:
    raw = str(exc)
    if not settings.semantic_scholar_api_key:
        return "当前未配置 Semantic Scholar API Key，匿名请求很容易被限流"
    if "rate limit" in raw.lower() or "429" in raw:
        return "Semantic Scholar 当前返回限流，请稍后重试"
    if "timeout" in raw.lower() or "timed out" in raw.lower() or "超时" in raw:
        return "Semantic Scholar 请求超时，请检查网络后重试"
    if "connection" in raw.lower() or "incompleteread" in raw.lower() or "连接中断" in raw:
        return "Semantic Scholar 连接中断，请稍后重试或减少返回数量"
    return f"Semantic Scholar 请求失败：{raw}"


def _semantic_scholar_error(exc: Exception) -> str:
    reason = _semantic_scholar_reason(exc)
    suggestion = (
        "已配置 SEMANTIC_SCHOLAR_API_KEY 时，请等待 1-2 秒后重试、减少连续刷新或降低返回数量；"
        if settings.semantic_scholar_api_key
        else "可配置 SEMANTIC_SCHOLAR_API_KEY 后重启服务再试；"
    )
    return (
        f"{reason}。顶会论文模式不会自动混入 arXiv；"
        f"{suggestion}也可暂用“arXiv 预印本”完成演示。"
    )


def _arxiv_error(exc: Exception) -> str:
    raw = str(exc)
    if "rate limit" in raw.lower() or "429" in raw:
        reason = "arXiv 当前限流，请稍后重试"
    elif "temporarily unavailable" in raw.lower() or "unavailable" in raw.lower():
        reason = "arXiv 暂时不可用"
    elif "timeout" in raw.lower() or "timed out" in raw.lower():
        reason = "arXiv 请求超时"
    else:
        reason = f"arXiv 获取失败：{raw}"
    return f"{reason}，可稍后重试或切换到顶会论文 / 顶会 + arXiv 模式。"


def _paper_date_key(paper) -> str:
    return paper.publication_date or (paper.published[:10] if paper.published else "") or str(paper.year or "")


def _sort_watch_papers(papers: list, sort_by: str) -> list:
    if sort_by == "latest":
        return sorted(papers, key=_paper_date_key, reverse=True)
    if sort_by == "citations":
        return sorted(
            papers,
            key=lambda paper: (paper.citation_count, paper.influential_citation_count, _paper_date_key(paper)),
            reverse=True,
        )
    return papers


def _search_watch_papers(topic, trace: AgentTrace) -> tuple[list, str, str]:
    source_warning = ""
    sort_warning = ""
    papers = []
    if topic.source_mode == "conference":
        if not settings.semantic_scholar_api_key:
            raise RuntimeError(_semantic_scholar_error(RuntimeError("missing api key")))
        try:
            papers = search_semantic_scholar(
                query=topic.query.strip(),
                max_results=topic.max_results,
                days=topic.days,
                venue_group=topic.venue_group,
                venues=topic.venues,
                sort_by=topic.sort_by,
            )
        except Exception as exc:
            raise RuntimeError(_semantic_scholar_error(exc)) from exc
    elif topic.source_mode == "hybrid":
        if settings.semantic_scholar_api_key:
            try:
                papers.extend(
                    search_semantic_scholar(
                        query=topic.query.strip(),
                        max_results=topic.max_results,
                        days=topic.days,
                        venue_group=topic.venue_group,
                        venues=topic.venues,
                        sort_by=topic.sort_by,
                    )
                )
            except Exception as exc:
                source_warning = f"顶会源暂未返回，已使用 arXiv 补充：{_semantic_scholar_reason(exc)}。"
                trace.add("Semantic Scholar 顶会检索受限", f"{topic.name}: {source_warning}", status="failed")
        else:
            source_warning = (
                "当前未配置 Semantic Scholar API Key，已按 hybrid 设置使用 arXiv 补充。"
                "拿到 key 后写入 .env 并重启服务即可启用顶会论文检索。"
            )
            trace.add("Semantic Scholar API Key 缺失", f"{topic.name}: {source_warning}", status="failed")
        try:
            papers.extend(_mark_arxiv_supplement(search_arxiv(topic.query.strip(), topic.max_results, topic.days)))
        except Exception as exc:
            arxiv_warning = _arxiv_error(exc)
            if papers:
                source_warning = f"{source_warning} {arxiv_warning}".strip()
                trace.add("arXiv 补充检索受限", f"{topic.name}: {arxiv_warning}", status="failed")
            else:
                raise RuntimeError(arxiv_warning) from exc
    else:
        try:
            papers = search_arxiv(topic.query.strip(), topic.max_results, topic.days)
        except Exception as exc:
            raise RuntimeError(_arxiv_error(exc)) from exc
        if topic.sort_by == "citations":
            sort_warning = "arXiv 不提供引用量，已按最新时间返回。"

    deduped = {}
    for paper in papers:
        key = _title_key(paper.title) or paper.id
        if key not in deduped:
            deduped[key] = paper
    deduped_papers = _sort_watch_papers(list(deduped.values()), topic.sort_by)
    if topic.source_mode == "hybrid" and topic.sort_by == "citations" and any(paper.source.startswith("arXiv") for paper in deduped_papers):
        sort_warning = "arXiv 补充论文没有引用量，引用量排序优先使用 Semantic Scholar 论文。"
    return deduped_papers[: topic.max_results], source_warning, sort_warning


def _ingest_pdf_url(pdf_url: str, title: str, trace: AgentTrace) -> UploadResponse:
    with trace.step("下载论文 PDF", pdf_url):
        pdf_file = download_pdf(pdf_url)
        pdf_bytes = pdf_file.getvalue()
    with trace.step("解析 PDF", title):
        pages, page_count, char_count = parse_pdf_bytes(pdf_bytes)
    if not pages:
        raise ValueError("No extractable text found in this PDF.")
    with trace.step("切分论文文本", f"{char_count} characters"):
        chunks = chunk_pages(pages)
    with trace.step("构建向量索引", f"{len(chunks)} chunks"):
        filename = f"{title[:80].strip() or 'arxiv_paper'}.pdf"
        paper = paper_store.add(filename, pages, chunks, pdf_bytes=pdf_bytes)
    return UploadResponse(
        paper_id=paper.paper_id,
        filename=filename,
        page_count=page_count,
        chunk_count=len(chunks),
        char_count=char_count,
        trace=trace.steps,
    )


def _ensure_visual_assets(paper_id: str) -> tuple[list[dict], dict[str, bytes]]:
    paper = paper_store.get(paper_id)
    if paper.visual_assets is not None and paper.visual_asset_images is not None:
        return paper.visual_assets, paper.visual_asset_images
    if not paper.pdf_bytes:
        paper_store.set_visual_assets(paper_id, [], {})
        return [], {}
    assets, images = extract_visual_assets(paper.pdf_bytes)
    paper_store.set_visual_assets(paper_id, assets, images)
    return assets, images


def _run_pdf_job(job_id: str, request: PdfUrlRequest) -> None:
    try:
        started = job_store.start_step(job_id, "下载 PDF", request.pdf_url)
        try:
            pdf_file = download_pdf(request.pdf_url)
            pdf_bytes = pdf_file.getvalue()
        except Exception as exc:
            job_store.fail_step(job_id, started, exc)
            return
        job_store.finish_step(job_id, started)

        started = job_store.start_step(job_id, "解析文本", request.title)
        try:
            pages, page_count, char_count = parse_pdf_bytes(pdf_bytes)
            if not pages:
                raise ValueError("No extractable text found in this PDF.")
        except Exception as exc:
            job_store.fail_step(job_id, started, exc)
            return
        job_store.finish_step(job_id, started)

        started = job_store.start_step(job_id, "切分 chunk", f"{char_count} characters")
        try:
            chunks = chunk_pages(pages)
        except Exception as exc:
            job_store.fail_step(job_id, started, exc)
            return
        job_store.finish_step(job_id, started)

        started = job_store.start_step(job_id, "构建向量索引", f"{len(chunks)} chunks")
        try:
            filename = f"{request.title[:80].strip() or 'arxiv_paper'}.pdf"
            paper = paper_store.add(filename, pages, chunks, pdf_bytes=pdf_bytes)
        except Exception as exc:
            job_store.fail_step(job_id, started, exc)
            return
        job_store.finish_step(job_id, started)

        result = UploadResponse(
            paper_id=paper.paper_id,
            filename=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            char_count=char_count,
            trace=job_store.get(job_id).trace,
        )
        job_store.complete(job_id, result.model_dump())
    except Exception as exc:
        started = job_store.start_step(job_id, "失败", request.title)
        job_store.fail_step(job_id, started, exc)


def _run_llm_job(job_id: str, paper_id: str, job_type: str) -> None:
    try:
        started = job_store.start_step(job_id, "读取论文内容", paper_id)
        try:
            paper = paper_store.get(paper_id)
        except Exception as exc:
            job_store.fail_step(job_id, started, exc)
            return
        job_store.finish_step(job_id, started)

        label = "结构化总结" if job_type == "summary" else "Reviewer 分析"
        visual_assets = []
        if job_type == "summary":
            started = job_store.start_step(job_id, "抽取图表资产", paper.filename)
            try:
                visual_assets, _ = _ensure_visual_assets(paper_id)
            except Exception as exc:
                job_store.fail_step(job_id, started, exc)
                return
            job_store.finish_step(job_id, started)
        started = job_store.start_step(job_id, f"调用 LLM 生成{label}", paper.filename)
        try:
            if job_type == "summary":
                system_prompt, user_prompt = summary_prompt(paper.filename, paper.chunks, visual_assets_prompt(visual_assets))
            else:
                system_prompt, user_prompt = reviewer_prompt(paper.filename, paper.chunks)
            content_parts = []
            for delta in stream_complete(system_prompt, user_prompt):
                content_parts.append(delta)
                job_store.append_content(job_id, delta)
            content = "".join(content_parts).strip()
            if not content:
                raise RuntimeError("LLM returned an empty response.")
        except Exception as exc:
            job_store.fail_step(job_id, started, exc)
            return
        job_store.finish_step(job_id, started)
        result = {"content": content}
        if job_type == "summary":
            result["visual_assets"] = visual_assets
        job_store.complete(job_id, result, content=content)
    except Exception as exc:
        started = job_store.start_step(job_id, "失败", paper_id)
        job_store.fail_step(job_id, started, exc)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model,
        embedding_model_loaded=model_loaded(),
        llm_configured=llm_configured(),
        semantic_scholar_configured=bool(settings.semantic_scholar_api_key),
    )


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    trace = AgentTrace()
    try:
        with trace.step("解析研究方向", request.query):
            query = request.query.strip()
        with trace.step("实时检索 arXiv", f"max_results={request.max_results}, days={request.days}"):
            papers = search_arxiv(query, request.max_results, request.days)
        trace.add("整理论文元信息", f"获取到 {len(papers)} 篇论文")
        return SearchResponse(papers=papers, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/rank", response_model=RankResponse)
def rank(request: RankRequest) -> RankResponse:
    trace = AgentTrace()
    try:
        with trace.step("计算论文 embedding", f"{len(request.papers)} 篇论文"):
            ranked = rank_papers(request.query, request.papers)
        trace.add("生成排序解释", "综合语义相关性、新近性、关键词匹配、代码线索和引用量")
        return RankResponse(papers=ranked, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/watch/refresh", response_model=WatchRefreshResponse)
def refresh_watch_topics(request: WatchRefreshRequest) -> WatchRefreshResponse:
    trace = AgentTrace()
    results: list[WatchTopicResult] = []
    for topic in request.topics:
        try:
            with trace.step("刷新关注领域", topic.name):
                papers, source_warning, sort_warning = _search_watch_papers(topic, trace)
            with trace.step("自动排序最新论文", f"{topic.name}: {len(papers)} papers"):
                ranked = rank_papers(topic.query, papers)
                ranked = _sort_watch_papers(ranked, topic.sort_by)
            seen_ids = set(topic.seen_ids)
            first_baseline = not topic.baseline_done and not seen_ids
            watch_papers = [
                WatchPaper(**paper.model_dump(), is_new=False if first_baseline else paper.id not in seen_ids)
                for paper in ranked
            ]
            results.append(
                WatchTopicResult(
                    id=topic.id,
                    name=topic.name,
                    query=topic.query,
                    source_mode=topic.source_mode,
                    venue_group=topic.venue_group,
                    venues=_topic_venues(topic),
                    checked_at=datetime.now(timezone.utc).isoformat(),
                    papers=watch_papers,
                    new_count=sum(1 for paper in watch_papers if paper.is_new),
                    total_count=len(watch_papers),
                    baseline_count=len(watch_papers) if first_baseline else 0,
                    status_message="刷新完成" if watch_papers else "无匹配论文",
                    source_warning=source_warning,
                    sort_by=topic.sort_by,
                    sort_warning=sort_warning,
                )
            )
        except Exception as exc:
            trace.add("关注领域刷新失败", f"{topic.name}: {exc}", status="failed")
            results.append(
                WatchTopicResult(
                    id=topic.id,
                    name=topic.name,
                    query=topic.query,
                    source_mode=topic.source_mode,
                    venue_group=topic.venue_group,
                    venues=_topic_venues(topic),
                    checked_at=datetime.now(timezone.utc).isoformat(),
                    papers=[],
                    new_count=0,
                    total_count=0,
                    baseline_count=0,
                    status_message="顶会检索失败" if topic.source_mode == "conference" else "刷新失败",
                    sort_by=topic.sort_by,
                    error=str(exc),
                )
            )
    return WatchRefreshResponse(topics=results, trace=trace.steps)


@app.post("/api/papers/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)) -> UploadResponse:
    trace = AgentTrace()
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        with trace.step("解析 PDF", file.filename):
            pdf_bytes = file.file.read()
            pages, page_count, char_count = parse_pdf_bytes(pdf_bytes)
        if not pages:
            raise ValueError("No extractable text found in this PDF.")
        with trace.step("切分论文文本", f"{char_count} characters"):
            chunks = chunk_pages(pages)
        with trace.step("构建向量索引", f"{len(chunks)} chunks"):
            paper = paper_store.add(file.filename, pages, chunks, pdf_bytes=pdf_bytes)
        return UploadResponse(
            paper_id=paper.paper_id,
            filename=file.filename,
            page_count=page_count,
            chunk_count=len(chunks),
            char_count=char_count,
            trace=trace.steps,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/papers/{paper_id}/pages/{page}/image")
def paper_page_image(paper_id: str, page: int, zoom: float = 1.6) -> Response:
    try:
        paper = paper_store.get(paper_id)
        if not paper.pdf_bytes:
            raise ValueError("This PDF was not stored with page image data. Please parse or upload it again.")
        if page < 1:
            raise ValueError("Page number must be greater than 0.")
        safe_zoom = min(max(zoom, 0.8), 2.2)
        with fitz.open(stream=paper.pdf_bytes, filetype="pdf") as document:
            if page > document.page_count:
                raise ValueError(f"Page {page} is outside this PDF. It has {document.page_count} pages.")
            matrix = fitz.Matrix(safe_zoom, safe_zoom)
            pixmap = document.load_page(page - 1).get_pixmap(matrix=matrix, alpha=False)
            return Response(
                content=pixmap.tobytes("png"),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=3600"},
            )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/papers/{paper_id}/visual-assets", response_model=VisualAssetsResponse)
def paper_visual_assets(paper_id: str) -> VisualAssetsResponse:
    trace = AgentTrace()
    try:
        with trace.step("抽取技术架构图和主结果表", paper_id):
            assets, _ = _ensure_visual_assets(paper_id)
        return VisualAssetsResponse(assets=assets, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/papers/{paper_id}/visual-assets/{asset_id}/image")
def paper_visual_asset_image(paper_id: str, asset_id: str) -> Response:
    try:
        _, images = _ensure_visual_assets(paper_id)
        image = images.get(asset_id)
        if not image:
            raise ValueError("Visual asset image not found. Try extracting visual assets again.")
        return Response(
            content=image,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/papers/from-url", response_model=UploadResponse)
def ingest_paper_from_url(request: PdfUrlRequest) -> UploadResponse:
    trace = AgentTrace()
    try:
        return _ingest_pdf_url(request.pdf_url, request.title, trace)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/papers/from-url/jobs", response_model=JobCreateResponse)
def create_pdf_job(request: PdfUrlRequest, background_tasks: BackgroundTasks) -> JobCreateResponse:
    job = job_store.create()
    background_tasks.add_task(_run_pdf_job, job.job_id, request)
    return JobCreateResponse(job_id=job.job_id, status=job.status, stage=job.stage)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    try:
        job = job_store.get(job_id)
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            error=job.error,
            result=job.result,
            content=job.content,
            trace=job.trace,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/api/llm-jobs/{job_id}/stream")
def stream_llm_job(job_id: str):
    def event_stream():
        sent = 0
        while True:
            try:
                job = job_store.get(job_id)
            except Exception as exc:
                yield _sse({"status": "failed", "stage": "失败", "error": str(exc), "delta": ""})
                return
            content = job.content or ""
            delta = content[sent:]
            sent = len(content)
            yield _sse(
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "stage": job.stage,
                    "error": job.error,
                    "delta": delta,
                    "content": content if job.status in {"completed", "failed"} else "",
                    "result": job.result if job.status in {"completed", "failed"} else None,
                    "trace": [step.model_dump() for step in job.trace],
                }
            )
            if job.status in {"completed", "failed"}:
                return
            time.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/papers/{paper_id}/summary", response_model=AnalysisResponse)
def paper_summary(paper_id: str) -> AnalysisResponse:
    trace = AgentTrace()
    try:
        with trace.step("读取论文索引", paper_id):
            paper = paper_store.get(paper_id)
        with trace.step("抽取图表资产", paper.filename):
            visual_assets, _ = _ensure_visual_assets(paper_id)
        with trace.step("调用 LLM 生成结构化总结", paper.filename):
            system_prompt, user_prompt = summary_prompt(paper.filename, paper.chunks, visual_assets_prompt(visual_assets))
            content = stream_complete(system_prompt, user_prompt)
            content = "".join(content).strip()
        return AnalysisResponse(content=content, visual_assets=visual_assets, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/papers/{paper_id}/summary/jobs", response_model=JobCreateResponse)
def create_summary_job(paper_id: str, background_tasks: BackgroundTasks) -> JobCreateResponse:
    job = job_store.create()
    background_tasks.add_task(_run_llm_job, job.job_id, paper_id, "summary")
    return JobCreateResponse(job_id=job.job_id, status=job.status, stage=job.stage)


@app.post("/api/papers/{paper_id}/qa", response_model=QAResponse)
def paper_qa(paper_id: str, request: QARequest) -> QAResponse:
    trace = AgentTrace()
    try:
        with trace.step("检索证据片段", request.question):
            paper = paper_store.get(paper_id)
            candidates = top_k(request.question, paper.chunks, max(request.top_k * 3, request.top_k))
            evidence = filter_evidence_chunks(candidates, request.top_k)
        with trace.step("调用 LLM 生成问答", f"{len(evidence)} evidence chunks"):
            answer = answer_question(request.question, evidence)
        return QAResponse(answer=answer, evidence=evidence, visual_pages=[], trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/papers/{paper_id}/reviewer", response_model=AnalysisResponse)
def paper_reviewer(paper_id: str) -> AnalysisResponse:
    trace = AgentTrace()
    try:
        with trace.step("读取论文内容", paper_id):
            paper = paper_store.get(paper_id)
        with trace.step("调用 LLM 生成 Reviewer 分析", paper.filename):
            content = reviewer_analysis(paper.filename, paper.chunks)
        return AnalysisResponse(content=content, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/papers/{paper_id}/reviewer/jobs", response_model=JobCreateResponse)
def create_reviewer_job(paper_id: str, background_tasks: BackgroundTasks) -> JobCreateResponse:
    job = job_store.create()
    background_tasks.add_task(_run_llm_job, job.job_id, paper_id, "reviewer")
    return JobCreateResponse(job_id=job.job_id, status=job.status, stage=job.stage)


@app.post("/api/trends", response_model=TrendsResponse)
def trends(request: TrendsRequest) -> TrendsResponse:
    trace = AgentTrace()
    try:
        with trace.step("聚类论文主题", f"{len(request.papers)} papers"):
            topics = build_topics(request.papers)
            counts = time_distribution(request.papers)
        with trace.step("调用 LLM 解释研究趋势", request.query):
            analysis = explain_trends(request.query, request.papers, topics, counts)
        return TrendsResponse(topics=topics, time_distribution=counts, llm_analysis=analysis, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/graph", response_model=GraphResponse)
def graph(request: GraphRequest) -> GraphResponse:
    trace = AgentTrace()
    try:
        with trace.step("计算论文关系图谱", f"threshold={request.similarity_threshold}"):
            nodes, edges = build_graph(request.papers, request.similarity_threshold)
        return GraphResponse(nodes=nodes, edges=edges, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/literature-review", response_model=AnalysisResponse)
def review(request: LiteratureReviewRequest) -> AnalysisResponse:
    trace = AgentTrace()
    try:
        with trace.step("整理多篇论文上下文", f"{len(request.papers)} papers"):
            papers = request.papers
        with trace.step("调用 LLM 生成文献综述", request.query):
            content = literature_review(request.query, papers, request.graph)
        return AnalysisResponse(content=content, trace=trace.steps)
    except Exception as exc:
        raise _http_error(exc) from exc
