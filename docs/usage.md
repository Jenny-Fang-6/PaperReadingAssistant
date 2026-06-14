# 使用说明

## 环境配置

项目需要 LLM API Key。复制 `.env.example` 为 `.env`，填写：

```text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
SEMANTIC_SCHOLAR_API_KEY 可选，申请期间可留空
```

Kimi / Moonshot 示例：

```text
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=moonshot-v1-8k
```

## 运行

推荐：

```bash
docker compose up --build
```

前端：`http://localhost:3000`

后端：`http://localhost:8000`

## 推荐使用流程

1. 在“论文追踪收件箱”中填写关注名称、研究关键词、论文来源、会议范围、返回数量、排序方式和时间范围，点击“添加并刷新”。
2. 申请 Semantic Scholar API Key 期间，页面默认使用 `arXiv 预印本`，可稳定演示最新论文追踪、自动获取元信息和工作台分析。
3. 拿到 Semantic Scholar API Key 后写入 `.env` 并重启 Docker，页面会默认使用 `顶会论文`。
4. `顶会论文` 只使用 Semantic Scholar 检索会议论文；遇到限流或超时会在对应卡片明确失败，不会自动混入 arXiv。
5. `arXiv 预印本` 只检索 arXiv；`顶会 + arXiv` 才允许在顶会源受限时使用 arXiv 补充，补充论文会标为“arXiv 补充”。
6. 首次刷新作为 baseline 收录，不把所有论文标为 `NEW`；后续刷新只标记真正新增的论文。
7. 点击“用于分析”或单篇论文的“加入工作台”，将论文送入下方分析工作台。
8. 左侧只保留排序、趋势、图谱、综述等全局分析操作；所有论文发现统一通过收件箱完成。
9. 点击论文卡片中的“解析 PDF”，在卡片中查看下载、解析文本、切分 chunk、构建向量索引等状态。
10. 解析完成后，到“单篇论文精读”继续生成结构化总结、Reviewer 分析和证据问答。
11. 结构化总结和 Reviewer 分析使用 Kimi/OpenAI-compatible API 流式生成，结果默认以 Markdown 渲染展示。

## 常见问题

- 如果 `/api/health` 显示 `llm_configured=false`，说明 `.env` 中缺少 `LLM_API_KEY`。
- 第一次启动会下载或加载 sentence-transformers 模型，耗时取决于网络；建议连接 VPN 后再启动 Docker。
- Docker Compose 会挂载 `${HOME}/.cache/huggingface` 作为模型缓存。
- 关注领域、已见论文和工作台保存在浏览器 localStorage 中；换浏览器或清空浏览器数据后需要重新添加。
- Semantic Scholar API 免费，但无 API Key 时容易限流；系统在未配置 key 时可以使用 arXiv，配置 `SEMANTIC_SCHOLAR_API_KEY` 后可稳定启用会议论文追踪。
- 会议范围支持 `全部会议`，也可选择 ML、NLP、AI、CV、IR/DM 等预设组。
- 排序方式包括 `最新优先`、`引用量优先`、`综合推荐`；arXiv 不提供引用量，选择引用量排序时会按最新时间返回并显示提示。
- 论文问答中的“依据片段”来自已解析 PDF 的本地向量检索，最终答案由 Kimi/OpenAI-compatible API 生成，不调用 Semantic Scholar 或外部搜索。
- 论文列表中的“解析 PDF”可以直接下载并解析 arXiv PDF，且会显示分步状态；如果网络失败，可手动下载后用“上传 PDF”继续精读。
- 单篇精读的图表辅助总结会复用 Kimi 的 `LLM_API_KEY` / `LLM_BASE_URL`，固定调用 `kimi-k2.6` 从本地候选裁剪图中选择技术路线图和主结果表；不需要额外图像模型 API，也不需要 `pdffigures2.jar`。如果 Kimi 没有选中明确图表，系统会显示低置信 caption 兜底候选，方便继续人工核对。
- 如果出现 Hugging Face、arXiv、Semantic Scholar timeout 或连接中断，优先检查是否已连接 VPN，尤其是在 Docker 容器中运行时。
- arXiv 请求失败通常是网络或频率限制问题，可以等待一段时间后重试。
- 扫描版 PDF 无法直接解析，需要先 OCR。
