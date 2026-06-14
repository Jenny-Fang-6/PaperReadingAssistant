import json
import re
import base64
from uuid import uuid4

import cv2
import fitz
import numpy as np

from .llm_client import LLMConfigurationError, vision_complete


FIGURE_WORDS = re.compile(r"\b(fig(?:ure)?\.?)\s*\d+|图\s*\d+", re.IGNORECASE)
TABLE_WORDS = re.compile(r"\btable\s*\d+|表\s*\d+", re.IGNORECASE)
ARCH_WORDS = re.compile(
    r"framework|architecture|overview|method|pipeline|model|system|workflow|approach|框架|架构|方法|流程|模型|系统",
    re.IGNORECASE,
)
MAX_KIMI_CANDIDATES = 12
MIN_CANDIDATE_AREA_RATIO = 0.018
MAX_CANDIDATE_AREA_RATIO = 0.62


def _clean_caption(value: str | None) -> str:
    return " ".join((value or "").split())[:900]


def _caption_score(caption: str, asset_type: str) -> int:
    score = 0
    if asset_type == "architecture":
        score += 5 if ARCH_WORDS.search(caption) else 0
        score += 2 if FIGURE_WORDS.search(caption) else 0
        score -= 4 if TABLE_WORDS.search(caption) else 0
    else:
        score += 5 if RESULT_WORDS.search(caption) else 0
        score += 2 if TABLE_WORDS.search(caption) else 0
        score -= 4 if FIGURE_WORDS.search(caption) else 0
    return score
RESULT_WORDS = re.compile(
    r"result|performance|comparison|ablation|experiment|evaluation|score|accuracy|benchmark|结果|性能|对比|消融|实验|评估",
    re.IGNORECASE,
)


def _block_text(block: dict) -> str:
    lines = block.get("lines", [])
    spans = [span.get("text", "") for line in lines for span in line.get("spans", [])]
    return " ".join(" ".join(spans).split())


def _caption_candidates(document: fitz.Document) -> list[dict]:
    candidates: list[dict] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            text = _block_text(block)
            if not text:
                continue
            is_figure = bool(FIGURE_WORDS.search(text))
            is_table = bool(TABLE_WORDS.search(text))
            if not is_figure and not is_table:
                continue
            if is_figure and not ARCH_WORDS.search(text):
                continue
            if is_table and not RESULT_WORDS.search(text):
                continue
            candidates.append(
                {
                    "page_index": page_index,
                    "page": page_index + 1,
                    "bbox": fitz.Rect(block["bbox"]),
                    "caption": text[:500],
                    "asset_type": "architecture" if is_figure else "result_table",
                }
            )
    return candidates


def _asset_label(asset_type: str, caption: str) -> str:
    if asset_type == "architecture":
        match = FIGURE_WORDS.search(caption)
        prefix = match.group(0).strip() if match else "技术架构图"
        return f"{prefix}：技术架构图"
    match = TABLE_WORDS.search(caption)
    prefix = match.group(0).strip() if match else "主结果表"
    return f"{prefix}：主结果表"


def _rect_to_bbox(rect: fitz.Rect) -> list[float]:
    return [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]


def _bbox_key(page: int, rect: fitz.Rect) -> tuple[int, int, int, int, int]:
    return (page, round(rect.x0 / 12), round(rect.y0 / 12), round(rect.x1 / 12), round(rect.y1 / 12))


def _nearby_caption(page: fitz.Page, rect: fitz.Rect) -> str:
    captions = []
    for block in page.get_text("dict").get("blocks", []):
        text = _block_text(block)
        if not text or not (FIGURE_WORDS.search(text) or TABLE_WORDS.search(text)):
            continue
        block_rect = fitz.Rect(block["bbox"])
        vertical_gap = min(abs(block_rect.y0 - rect.y1), abs(rect.y0 - block_rect.y1))
        horizontal_overlap = min(block_rect.x1, rect.x1) - max(block_rect.x0, rect.x0)
        if vertical_gap <= page.rect.height * 0.18 and horizontal_overlap > -page.rect.width * 0.08:
            captions.append((vertical_gap, text))
    return _clean_caption(sorted(captions, key=lambda item: item[0])[0][1]) if captions else ""


def _candidate_type(caption: str, rect: fitz.Rect, page_rect: fitz.Rect) -> str:
    if TABLE_WORDS.search(caption):
        return "table_caption"
    if FIGURE_WORDS.search(caption):
        return "figure_caption"
    if rect.width > page_rect.width * 0.62 and rect.height < page_rect.height * 0.38:
        return "wide_region"
    return "visual_region"


def _render_clip(page: fitz.Page, rect: fitz.Rect, zoom: float = 1.7) -> bytes:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return pixmap.tobytes("png")


def _add_candidate(
    candidates: list[dict],
    images: dict[str, bytes],
    seen: set[tuple[int, int, int, int, int]],
    page: fitz.Page,
    page_number: int,
    rect: fitz.Rect,
    candidate_type: str,
    caption: str = "",
    score: float = 0.0,
) -> None:
    clipped = _clip_to_page(_expand(rect, 8, 8), page.rect)
    area_ratio = (clipped.width * clipped.height) / max(page.rect.width * page.rect.height, 1)
    if area_ratio < MIN_CANDIDATE_AREA_RATIO or area_ratio > MAX_CANDIDATE_AREA_RATIO:
        return
    key = _bbox_key(page_number, clipped)
    if key in seen:
        return
    seen.add(key)
    asset_id = str(uuid4())
    try:
        image_bytes = _render_clip(page, clipped)
    except Exception:
        return
    images[asset_id] = image_bytes
    candidates.append(
        {
            "candidate_id": asset_id,
            "page": page_number,
            "bbox": _rect_to_bbox(clipped),
            "candidate_type": candidate_type,
            "caption_nearby": caption,
            "score": score,
            "area_ratio": round(area_ratio, 4),
        }
    )


def _cv_region_candidates(page: fitz.Page, page_number: int) -> list[tuple[fitz.Rect, float]]:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, pixmap.width // 28), 2))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(18, pixmap.height // 35)))
    line_mask = cv2.bitwise_or(cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel), cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel))
    visual_mask = cv2.dilate(cv2.bitwise_or(binary, line_mask), cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=2)
    contours, _ = cv2.findContours(visual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_x = page.rect.width / max(pixmap.width, 1)
    scale_y = page.rect.height / max(pixmap.height, 1)
    rects: list[tuple[fitz.Rect, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / max(pixmap.width * pixmap.height, 1)
        if area_ratio < MIN_CANDIDATE_AREA_RATIO or area_ratio > MAX_CANDIDATE_AREA_RATIO:
            continue
        if w < pixmap.width * 0.16 or h < pixmap.height * 0.045:
            continue
        rect = fitz.Rect(x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y)
        line_density = float(cv2.countNonZero(line_mask[y : y + h, x : x + w])) / max(w * h, 1)
        score = area_ratio * 100 + line_density * 70
        rects.append((rect, score))
    return sorted(rects, key=lambda item: item[1], reverse=True)[:18]


def _local_visual_candidates(pdf_bytes: bytes, max_candidates: int = MAX_KIMI_CANDIDATES) -> tuple[list[dict], dict[str, bytes]]:
    candidates: list[dict] = []
    images: dict[str, bytes] = {}
    seen: set[tuple[int, int, int, int, int]] = set()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        caption_items = _caption_candidates(document)
        for candidate in caption_items:
            page = document.load_page(candidate["page_index"])
            clip = _crop_rect(page, candidate["bbox"], candidate["asset_type"])
            if clip is None:
                continue
            _add_candidate(
                candidates,
                images,
                seen,
                page,
                candidate["page"],
                clip,
                f"{candidate['asset_type']}_caption",
                candidate["caption"],
                score=80 + _caption_score(candidate["caption"], candidate["asset_type"]),
            )

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            for rect, score in _cv_region_candidates(page, page_number):
                caption = _nearby_caption(page, rect)
                candidate_type = _candidate_type(caption, rect, page.rect)
                _add_candidate(candidates, images, seen, page, page_number, rect, candidate_type, caption, score)

            if page_index < min(document.page_count, 4):
                middle = fitz.Rect(page.rect.width * 0.04, page.rect.height * 0.12, page.rect.width * 0.96, page.rect.height * 0.88)
                _add_candidate(candidates, images, seen, page, page_number, middle, "page_region", _nearby_caption(page, middle), score=8)

    selected = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_candidates]
    selected_ids = {item["candidate_id"] for item in selected}
    return selected, {asset_id: image for asset_id, image in images.items() if asset_id in selected_ids}


def _image_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _clean_json_response(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Vision model did not return a JSON object.")
    return data


def _kimi_select_assets(candidates: list[dict], candidate_images: dict[str, bytes], max_assets: int = 2) -> tuple[list[dict], dict[str, bytes]]:
    if not candidates:
        return [], {}
    prompt_candidates = [
        {
            "candidate_id": candidate["candidate_id"],
            "image_index": index + 1,
            "page": candidate["page"],
            "bbox": candidate["bbox"],
            "candidate_type": candidate["candidate_type"],
            "caption_nearby": candidate.get("caption_nearby", ""),
        }
        for index, candidate in enumerate(candidates)
    ]
    images = [
        {"data_url": _image_data_url(candidate_images[candidate["candidate_id"]]), "detail": "low"}
        for candidate in candidates
        if candidate["candidate_id"] in candidate_images
    ]
    if not images:
        return [], {}
    prompt = (
        "请从候选论文截图中选择最多两张图表：一张技术路线图/方法框架图，一张主结果表。"
        "只能选择候选列表中的 candidate_id，不能编造不存在的图表。"
        "如果候选图只是正文、摘要页、参考文献、普通段落或不明确，请返回 null。"
        "技术路线图通常展示 architecture/framework/pipeline/model/workflow/method overview；"
        "主结果表通常展示实验指标、baseline 对比、ablation、performance 或 benchmark results。"
        "请返回严格 JSON 对象，不要 Markdown，格式为："
        "{\"architecture_candidate_id\": string|null, \"result_table_candidate_id\": string|null, "
        "\"confidence\": \"high|medium|low\", \"reason\": string, \"reject_reason\": string}。\n\n"
        f"候选列表：{json.dumps(prompt_candidates, ensure_ascii=False)}"
    )
    try:
        content = vision_complete("你是学术 PDF 图表判别助手，只输出 JSON。", prompt, images)
        data = _clean_json_response(content)
    except (LLMConfigurationError, Exception):
        return [], {}

    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    selected_specs = [
        ("architecture", data.get("architecture_candidate_id")),
        ("result_table", data.get("result_table_candidate_id")),
    ]
    confidence = str(data.get("confidence") or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    if confidence == "low":
        return [], {}

    assets: list[dict] = []
    images_out: dict[str, bytes] = {}
    reason = str(data.get("reason") or "").strip()
    for asset_type, candidate_id in selected_specs:
        if not candidate_id or candidate_id not in by_id or candidate_id not in candidate_images:
            continue
        candidate = by_id[candidate_id]
        caption = _clean_caption(candidate.get("caption_nearby", ""))
        images_out[candidate_id] = candidate_images[candidate_id]
        assets.append(
            {
                "asset_id": candidate_id,
                "asset_type": asset_type,
                "page": candidate["page"],
                "label": _asset_label(asset_type, caption),
                "caption": caption,
                "reason": reason or ("Kimi k2.6 判断该候选更符合技术路线图。" if asset_type == "architecture" else "Kimi k2.6 判断该候选更符合主结果表。"),
                "source": "kimi-selected",
                "confidence": confidence,
                "bbox": candidate["bbox"],
                "selection_source": "kimi-k2.6",
                "vision_reason": reason,
                "candidate_count": len(candidates),
            }
        )
        if len(assets) >= max_assets:
            break
    return assets, images_out


def _clip_to_page(rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        max(0, rect.x0),
        max(0, rect.y0),
        min(page_rect.width, rect.x1),
        min(page_rect.height, rect.y1),
    )


def _expand(rect: fitz.Rect, x: float = 12, y: float = 10) -> fitz.Rect:
    return fitz.Rect(rect.x0 - x, rect.y0 - y, rect.x1 + x, rect.y1 + y)


def _union(first: fitz.Rect, second: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        min(first.x0, second.x0),
        min(first.y0, second.y0),
        max(first.x1, second.x1),
        max(first.y1, second.y1),
    )


def _table_rect_from_detector(page: fitz.Page, caption_rect: fitz.Rect) -> fitz.Rect | None:
    if not hasattr(page, "find_tables"):
        return None
    try:
        finder = page.find_tables()
        tables = getattr(finder, "tables", []) or []
    except Exception:
        return None

    best: tuple[float, fitz.Rect] | None = None
    for table in tables:
        try:
            table_rect = fitz.Rect(table.bbox)
        except Exception:
            continue
        vertical_gap = min(abs(table_rect.y1 - caption_rect.y0), abs(table_rect.y0 - caption_rect.y1))
        if vertical_gap > page.rect.height * 0.18:
            continue
        overlap = min(table_rect.x1, caption_rect.x1) - max(table_rect.x0, caption_rect.x0)
        if overlap <= 0:
            continue
        score = vertical_gap - overlap / max(page.rect.width, 1)
        if best is None or score < best[0]:
            best = (score, _union(table_rect, caption_rect))
    return best[1] if best else None


def _block_table_score(text: str) -> float:
    if not text:
        return 0.0
    number_count = len(re.findall(r"\d+(?:\.\d+)?", text))
    token_count = len(re.findall(r"\S+", text))
    separator_count = text.count("|") + text.count("✓") + text.count("×")
    return number_count * 2.0 + separator_count * 1.5 + min(token_count, 40) * 0.08


def _table_rect_from_text(page: fitz.Page, caption_rect: fitz.Rect) -> fitz.Rect | None:
    page_rect = page.rect
    blocks = []
    for block in page.get_text("dict").get("blocks", []):
        text = _block_text(block)
        if not text:
            continue
        rect = fitz.Rect(block["bbox"])
        if rect.intersects(caption_rect):
            continue
        if abs(rect.y0 - caption_rect.y0) > page_rect.height * 0.42 and abs(rect.y1 - caption_rect.y1) > page_rect.height * 0.42:
            continue
        score = _block_table_score(text)
        if score <= 0:
            continue
        blocks.append((rect, score))

    above = [(rect, score) for rect, score in blocks if rect.y1 <= caption_rect.y0 + 4 and caption_rect.y0 - rect.y0 < page_rect.height * 0.36]
    below = [(rect, score) for rect, score in blocks if rect.y0 >= caption_rect.y1 - 4 and rect.y1 - caption_rect.y1 < page_rect.height * 0.36]
    above_score = sum(score for _, score in above)
    below_score = sum(score for _, score in below)
    chosen = above if above_score >= below_score else below
    if not chosen or max(above_score, below_score) < 22:
        return None

    # Keep only the dense table neighborhood closest to the caption. This avoids
    # swallowing the body text that often follows a table caption.
    if chosen is above:
        chosen = sorted(chosen, key=lambda item: caption_rect.y0 - item[0].y1)[:8]
    else:
        chosen = sorted(chosen, key=lambda item: item[0].y0 - caption_rect.y1)[:8]
    rect = caption_rect
    for block_rect, _ in chosen:
        rect = _union(rect, block_rect)
    return rect


def _crop_rect(page: fitz.Page, caption_rect: fitz.Rect, asset_type: str) -> fitz.Rect | None:
    rect = page.rect
    x0 = max(0, min(caption_rect.x0 - 24, rect.width * 0.04))
    x1 = min(rect.width, max(caption_rect.x1 + 24, rect.width * 0.96))
    if asset_type == "architecture":
        y0 = max(0, caption_rect.y0 - rect.height * 0.42)
        y1 = min(rect.height, caption_rect.y1 + 28)
    else:
        table_rect = _table_rect_from_detector(page, caption_rect) or _table_rect_from_text(page, caption_rect)
        if table_rect:
            crop = _clip_to_page(_expand(table_rect, 18, 14), rect)
            if crop.height > rect.height * 0.5:
                center = (caption_rect.y0 + caption_rect.y1) / 2
                crop.y0 = max(0, center - rect.height * 0.24)
                crop.y1 = min(rect.height, center + rect.height * 0.24)
            return crop
        return None
    # Never return a paper-sized screenshot; if the heuristic becomes too broad,
    # shrink it toward the caption neighborhood.
    if y1 - y0 > rect.height * 0.55:
        center = (y0 + y1) / 2
        half = rect.height * 0.275
        y0 = max(0, center - half)
        y1 = min(rect.height, center + half)
    return _clip_to_page(fitz.Rect(x0, y0, x1, y1), rect)


def extract_visual_assets(pdf_bytes: bytes, max_assets: int = 2) -> tuple[list[dict], dict[str, bytes]]:
    if not pdf_bytes:
        return [], {}
    local_candidates, local_images = _local_visual_candidates(pdf_bytes)
    kimi_assets, kimi_images = _kimi_select_assets(local_candidates, local_images, max_assets)
    if kimi_assets:
        return kimi_assets, kimi_images

    assets: list[dict] = []
    images: dict[str, bytes] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        candidates = _caption_candidates(document)
        selected: list[dict] = []
        for asset_type in ("architecture", "result_table"):
            match = next((candidate for candidate in candidates if candidate["asset_type"] == asset_type), None)
            if match:
                selected.append(match)
        for candidate in selected[:max_assets]:
            page = document.load_page(candidate["page_index"])
            clip = _crop_rect(page, candidate["bbox"], candidate["asset_type"])
            if clip is None:
                continue
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
            asset_id = str(uuid4())
            images[asset_id] = pixmap.tobytes("png")
            assets.append(
                {
                    "asset_id": asset_id,
                    "asset_type": candidate["asset_type"],
                    "page": candidate["page"],
                    "label": _asset_label(candidate["asset_type"], candidate["caption"]),
                    "caption": candidate["caption"],
                    "reason": (
                        "caption 指向方法框架、系统结构或流程图。"
                        if candidate["asset_type"] == "architecture"
                        else "caption 指向结果、性能对比或实验评估表。"
                    ),
                    "source": "pymupdf_fallback",
                    "confidence": "low",
                    "bbox": _rect_to_bbox(clip),
                    "selection_source": "fallback",
                    "vision_reason": "",
                    "candidate_count": len(local_candidates),
                }
            )
    return assets, images


def visual_assets_prompt(assets: list[dict]) -> str:
    if not assets:
        return "未识别到明确的技术架构图或主结果表；总结时不要声称已经读取这些图表。"
    lines = []
    for asset in assets:
        lines.append(
            f"- {asset.get('label')}，第 {asset.get('page')} 页，来源：{asset.get('source', 'unknown')}，"
            f"置信度：{asset.get('confidence', 'unknown')}，caption：{asset.get('caption')}"
        )
    if not lines:
        return "未识别到明确的技术架构图或主结果表；总结时不要声称已经读取这些图表。"
    return (
        "已从 PDF 中识别到以下图表资产。若置信度为 low，请在总结中明确说明这是低置信候选，"
        "只做辅助参考，不要把图表内容当作确定事实：\n" + "\n".join(lines)
    )
