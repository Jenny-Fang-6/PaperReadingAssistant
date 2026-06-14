from collections import Counter, defaultdict
import json
import re

import numpy as np

from .embeddings import cosine_similarity_matrix, encode_texts
from .llm_client import complete
from .schemas import GraphEdge, GraphNode, Paper, RankedPaper, TrendTopic


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "using",
    "based",
    "large",
    "model",
    "models",
    "learning",
    "paper",
    "study",
}


def evidence_prompt(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[依据 {index + 1} | Page {chunk['page']}, score={chunk.get('score', 0):.3f}]\n{chunk['text']}"
        for index, chunk in enumerate(chunks)
    )


def summary_prompt(filename: str, chunks: list[dict], visual_context: str = "") -> tuple[str, str]:
    context = "\n\n".join(chunk["text"] for chunk in chunks[:12])
    visual_instruction = (
        f"\n\n图表资产信息：\n{visual_context}\n"
        "如果有明确图表资产，请在结构化总结中加入“图表辅助解读”小节，分别解释技术架构图和主结果表如何支持论文结论。"
        "如果未识别到明确图表，不要编造图表。"
        if visual_context
        else ""
    )
    return (
        "你是严谨的科研论文阅读助手，请用中文输出结构化 Markdown 分析。",
        (
            f"请基于以下论文内容为 {filename} 生成结构化总结，必须包含："
            "研究背景、研究问题、核心方法、实验设置、主要结果、创新点分析、局限性、复现建议、扩展方向。"
            "其中“创新点分析”必须写得像人工论文阅读笔记，不要使用“新在哪里/为什么重要/论文中的证据”这类固定模板小标题。"
            "请从问题定义或研究视角、方法/模型结构、训练或推理流程、实验验证方式、应用价值等角度中选择真正相关的角度展开；"
            "每个角度先用一段自然、准确的中文分析说明论文相对已有工作的具体新意和意义，"
            "再用一句“论文依据：...”简短指出文中支撑该判断的设计、实验或结果。"
            "如果某个角度不适用于论文，可以省略或合并，不要为了凑格式硬写。"
            f"{visual_instruction}\n\n"
            f"{context}"
        ),
    )


def summarize_pdf(filename: str, chunks: list[dict]) -> str:
    system_prompt, user_prompt = summary_prompt(filename, chunks)
    return complete(system_prompt, user_prompt)


def answer_question(question: str, chunks: list[dict]) -> str:
    return complete(
        "你是基于证据回答的论文问答助手。只能根据给定片段回答，不确定时说明不足。",
        (
            f"问题：{question}\n\n依据片段：\n{evidence_prompt(chunks)}\n\n"
            "请用中文自然回答，不要输出“依据 1/2/3”的编号式引用。"
            "如果问题涉及图表或实验结果，但片段不足以确认，请明确说明需要查看结构化总结中的图表辅助解读或原论文。"
            "不要声称使用了外部检索或参考文献，不要伪造原论文图片或表格。"
        ),
    )


def extract_visual_references(question: str, answer: str, chunks: list[dict], max_pages: int = 3) -> list[dict]:
    visual_pattern = re.compile(r"\b(fig(?:ure)?\.?|table|algorithm)\s*[\dIVXivx.-]*|图\s*\d+|表\s*\d+|算法\s*\d+", re.IGNORECASE)
    question_wants_visuals = bool(re.search(r"图|表|框架|架构|结果|实验|对比|figure|table|architecture|framework|result", question, re.IGNORECASE))
    candidates = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if visual_pattern.search(text) or visual_pattern.search(answer) or question_wants_visuals:
            candidates.append({
                "page": chunk.get("page"),
                "score": round(float(chunk.get("score", 0)), 3),
                "text": text[:1400],
            })
    if not candidates:
        return []

    prompt = (
        "你要判断论文问答中是否值得展示原论文页面截图。"
        "请只从候选 PDF 片段中选择最多 3 个页面，优先选择包含 Figure/Table/Algorithm/图/表，"
        "或对问题答案最有帮助的结果表、框架图所在页。"
        "返回严格 JSON 数组，每项包含 page、label、reason。"
        "label 使用简短中文，例如“表 1：实验结果”或“图 2：方法框架”。"
        "reason 说明为什么这页有助于回答问题。不要返回 Markdown 代码块。\n\n"
        f"问题：{question}\n\n回答：{answer}\n\n候选片段：\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    try:
        content = complete("你是论文图表页选择助手，只输出 JSON。", prompt)
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        data = json.loads(cleaned)
        pages: list[dict] = []
        seen_pages: set[int] = set()
        for item in data if isinstance(data, list) else []:
            try:
                page = int(item.get("page"))
            except Exception:
                continue
            if page < 1 or page in seen_pages:
                continue
            seen_pages.add(page)
            pages.append({
                "page": page,
                "label": str(item.get("label") or f"第 {page} 页").strip()[:80],
                "reason": str(item.get("reason") or "").strip()[:180],
            })
            if len(pages) >= max_pages:
                break
        if pages:
            return pages
    except Exception:
        pass

    fallback_pages: list[dict] = []
    seen_pages: set[int] = set()
    for chunk in candidates:
        page = int(chunk.get("page") or 0)
        if page < 1 or page in seen_pages:
            continue
        seen_pages.add(page)
        fallback_pages.append({
            "page": page,
            "label": f"第 {page} 页",
            "reason": "该页包含与回答相关的图表或方法/实验线索，可查看原论文页面确认细节。",
        })
        if len(fallback_pages) >= max_pages:
            break
    return fallback_pages


def refine_evidence_chunks(question: str, answer: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    payload = [
        {"index": index + 1, "page": chunk.get("page"), "score": chunk.get("score", 0), "text": chunk.get("text", "")[:1800]}
        for index, chunk in enumerate(chunks)
    ]
    prompt = (
        "你要为论文问答整理可读的依据索引。请只根据给定 PDF 片段工作，不要补充外部知识。"
        "对每个依据片段，抽取或改写成 1-2 句与问题和回答最相关的中文依据说明；"
        "如果片段明显是目录、表格残片、参考文献或与问题弱相关，请写“该片段相关性较弱：...”并简述原因。"
        "必须返回严格 JSON 数组，每项包含 index、page、evidence 三个字段，不要包裹 Markdown 代码块。\n\n"
        f"问题：{question}\n\n回答：{answer}\n\n候选片段 JSON：\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        content = complete("你是论文证据整理助手，只输出 JSON。", prompt)
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        data = json.loads(cleaned)
        by_index = {int(item.get("index", 0)): item for item in data if isinstance(item, dict)}
        refined: list[dict] = []
        for index, chunk in enumerate(chunks, start=1):
            item = by_index.get(index, {})
            text = str(item.get("evidence") or "").strip()
            refined.append({
                **chunk,
                "page": int(item.get("page") or chunk.get("page") or 0),
                "text": text or chunk.get("text", ""),
            })
        return refined
    except Exception:
        return chunks


def _is_noisy_evidence(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) < 90:
        return True
    lower = compact.lower()
    if re.search(r"\b(contents|table of contents|references|bibliography|appendix outline)\b", lower):
        return True
    if compact.count(".") > max(35, len(compact) * 0.12):
        return True
    alpha_count = sum(char.isalpha() for char in compact)
    return alpha_count / max(len(compact), 1) < 0.45


def filter_evidence_chunks(chunks: list[dict], limit: int) -> list[dict]:
    filtered = [chunk for chunk in chunks if not _is_noisy_evidence(chunk.get("text", ""))]
    return (filtered or chunks)[:limit]


def reviewer_analysis(filename: str, chunks: list[dict]) -> str:
    system_prompt, user_prompt = reviewer_prompt(filename, chunks)
    return complete(system_prompt, user_prompt)


def reviewer_prompt(filename: str, chunks: list[dict]) -> tuple[str, str]:
    context = "\n\n".join(chunk["text"] for chunk in chunks[:14])
    return (
        "你是一名严格但建设性的学术会议 reviewer。",
        (
            f"请从 Reviewer 视角用中文分析论文 {filename}。请使用 Markdown 输出，必要的英文技术术语可以保留原文，"
            "但标题和主要解释必须是中文。必须包含这些小节：总体评价、主要优点、主要不足、"
            "给作者的问题、缺失实验、潜在风险、改进建议、可复现性分析。"
            "不要把整段内容放进 Markdown 代码块，也不要给每行添加代码缩进。\n\n"
            f"{context}"
        ),
    )


def _keywords(text: str, limit: int = 4) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
        if word.lower() not in STOPWORDS
    ]
    return [word for word, _ in Counter(words).most_common(limit)]


def build_topics(papers: list[Paper], target_topics: int = 5) -> list[TrendTopic]:
    if not papers:
        return []
    texts = [f"{paper.title}. {paper.summary}" for paper in papers]
    vectors = encode_texts(texts)
    similarity = cosine_similarity_matrix(vectors)
    remaining = set(range(len(papers)))
    topics: list[TrendTopic] = []

    while remaining and len(topics) < target_topics:
        seed = max(remaining, key=lambda idx: float(similarity[idx].sum()))
        members = [idx for idx in list(remaining) if similarity[seed, idx] >= 0.42 or idx == seed]
        for idx in members:
            remaining.discard(idx)
        combined = " ".join(texts[idx] for idx in members)
        keywords = _keywords(combined, 3)
        topics.append(
            TrendTopic(
                name=" / ".join(keywords) if keywords else f"Topic {len(topics) + 1}",
                paper_count=len(members),
                paper_ids=[papers[idx].id for idx in members],
            )
        )

    if remaining:
        members = sorted(remaining)
        combined = " ".join(texts[idx] for idx in members)
        keywords = _keywords(combined, 3)
        topics.append(
            TrendTopic(
                name=" / ".join(keywords) if keywords else "Other",
                paper_count=len(members),
                paper_ids=[papers[idx].id for idx in members],
            )
        )
    return topics


def time_distribution(papers: list[Paper]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for paper in papers:
        key = paper.published[:7] if paper.published else "unknown"
        counts[key] += 1
    return dict(sorted(counts.items()))


def explain_trends(query: str, papers: list[Paper], topics: list[TrendTopic], counts: dict[str, int]) -> str:
    paper_lines = "\n".join(f"- {paper.title}: {paper.summary[:700]}" for paper in papers[:18])
    topic_lines = "\n".join(f"- {topic.name}: {topic.paper_count} papers" for topic in topics)
    return complete(
        "你是科研趋势分析助手，擅长从最新论文中归纳主题、方法演进和研究机会。",
        (
            f"研究方向：{query}\n\n主题聚类：\n{topic_lines}\n\n时间分布：{counts}\n\n论文摘要：\n{paper_lines}\n\n"
            "请输出中文趋势分析，包含：热门主题、方法趋势、任务趋势、代表论文、潜在研究机会。"
        ),
    )


def build_graph(papers: list[Paper], threshold: float) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes = [
        GraphNode(
            id=paper.id,
            title=paper.title,
            category=paper.categories[0] if paper.categories else "",
            published=paper.published,
        )
        for paper in papers
    ]
    if len(papers) < 2:
        return nodes, []
    vectors = encode_texts([f"{paper.title}. {paper.summary}" for paper in papers])
    similarity = cosine_similarity_matrix(vectors)
    edges: list[GraphEdge] = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            weight = float(similarity[i, j])
            if weight >= threshold:
                shared = sorted(set(_keywords(papers[i].summary, 8)) & set(_keywords(papers[j].summary, 8)))
                reason = f"摘要语义相似度 {weight:.2f}"
                if shared:
                    reason += f"，共享关键词：{', '.join(shared[:4])}"
                edges.append(GraphEdge(source=papers[i].id, target=papers[j].id, weight=round(weight, 4), reason=reason))
    return nodes, sorted(edges, key=lambda edge: edge.weight, reverse=True)


def literature_review(query: str, papers: list[Paper], graph: dict | None) -> str:
    paper_lines = "\n".join(
        f"- {idx + 1}. {paper.title} ({paper.published[:10]}, score={getattr(paper, 'score', 'unranked')}): {paper.summary[:900]}"
        for idx, paper in enumerate(papers[:15])
    )
    edge_lines = ""
    if graph:
        edge_lines = "\n".join(
            f"- {edge.get('source')} -> {edge.get('target')}: {edge.get('weight')}"
            for edge in graph.get("edges", [])[:20]
        )
    return complete(
        "你是学术综述写作助手，输出应有逻辑结构、引用代表论文标题，并避免编造未给出的信息。",
        (
            f"研究方向：{query}\n\n候选论文：\n{paper_lines}\n\n论文关系：\n{edge_lines}\n\n"
            "请生成中文文献综述，包含：研究背景、主题脉络、代表工作、方法演进、现有局限、未来研究方向。"
        ),
        temperature=0.3,
    )
