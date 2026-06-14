# 项目设计说明

## 定位

论文阅读助手不是单点 PDF 总结器，而是面向科研流程的 Agent 式学术助手。系统覆盖关注领域追踪、论文发现、筛选、精读、问答、评价和沉淀。

## 架构

```text
React 前端
  ↓
FastAPI 后端
  ↓
Semantic Scholar 顶会检索 / arXiv 预印本补充 / PDF 解析 / sentence-transformers embedding / OpenAI 兼容 LLM
```

## 核心模块

- 论文追踪收件箱：前端用 localStorage 保存关注方向、论文来源、会议范围、排序方式、已见论文 ID 和工作台，打开页面或手动刷新时自动获取最新论文。
- 顶会检索：`顶会论文` 模式只调用 Semantic Scholar Graph API 获取 venue、year、publication date、PDF 链接等元信息；限流或超时时在 topic 卡片明确失败，不自动 fallback 到 arXiv。未配置 API Key 时，前端默认使用 `arXiv 预印本`，申请期仍可稳定演示。
- 预印本检索：`arXiv 预印本` 模式只查 arXiv；`顶会 + arXiv` 模式允许 arXiv 作为补充，并在结果来源中明确标为“arXiv 补充”。
- 统一发现入口：左侧不再提供临时检索输入，所有论文发现都通过收件箱表单完成。
- 分析工作台：关注区只展示精简收件箱，用户点击“用于分析”或“加入工作台”后，论文才进入排序、图谱、趋势和综述流程。
- 论文排序：支持最新优先、引用量优先和综合推荐；综合推荐结合语义相关性、新近性、关键词匹配、代码线索和引用量。
- PDF 精读：支持在线 PDF URL 或本地上传，用 pypdf 抽取文本，按页切分 chunk，并构建内存向量索引。
- PDF 任务状态：在线 PDF 解析使用后端内存任务队列，前端轮询展示下载、解析文本、切分 chunk、构建向量索引和失败原因。
- RAG 问答：用 sentence-transformers 在已解析 PDF chunk 中检索依据片段，再交给 LLM 生成答案；证据片段不是外部参考文献引用。
- LLM 流式生成：结构化总结和 Reviewer 分析通过后台任务调用当前 `.env` 配置的 OpenAI-compatible API，并用 SSE 向前端流式返回文本。
- Markdown 阅读：总结、Reviewer、问答、趋势和综述统一支持 Markdown 渲染与源码切换。
- 趋势分析：对多篇论文做语义聚类，并调用 LLM 解释研究趋势。
- 论文关系分析：基于摘要 embedding 相似度连接论文节点，展示主题或方法路线相近的论文对，不代表真实引用网络。
- 文献综述：基于多篇论文摘要、排序结果和图谱关系生成综述。
- Agent 轨迹：记录每个任务步骤的状态、耗时和输出摘要。
