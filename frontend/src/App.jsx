import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const WATCH_STORAGE_KEY = 'paper-reading-assistant.watch-fields.v1'
const LEGACY_WATCH_STORAGE_KEY = 'paper-reading-assistant.watch-topics.v6'
const OLDER_WATCH_STORAGE_KEY = 'paper-reading-assistant.watch-topics.v5'
const WORKBENCH_STORAGE_KEY = 'paper-reading-assistant.workbench.v3'
const LEGACY_WORKBENCH_STORAGE_KEY = 'paper-reading-assistant.workbench.v2'
const OLDER_WORKBENCH_STORAGE_KEY = 'paper-reading-assistant.workbench.v1'
const ANALYSIS_CACHE_STORAGE_KEY = 'paper-reading-assistant.analysis-cache.v1'
const READING_RECORD_STORAGE_KEY = 'paper-reading-assistant.reading-records.v1'
const QA_HISTORY_STORAGE_KEY = 'paper-reading-assistant.qa-history.v1'
const VENUE_GROUPS = {
  ALL: ['NeurIPS', 'ICML', 'ICLR', 'ACL', 'EMNLP', 'NAACL', 'COLING', 'AAAI', 'IJCAI', 'CVPR', 'ICCV', 'ECCV', 'SIGIR', 'KDD', 'WWW'],
  ML: ['NeurIPS', 'ICML', 'ICLR'],
  NLP: ['ACL', 'EMNLP', 'NAACL', 'COLING'],
  AI: ['AAAI', 'IJCAI'],
  CV: ['CVPR', 'ICCV', 'ECCV'],
  'IR/DM': ['SIGIR', 'KDD', 'WWW'],
}
const FIELD_PRESETS = {
  CUSTOM: { label: '自定义领域', venueGroup: 'ALL', keywords: '' },
  CV: { label: '计算机视觉', venueGroup: 'CV', keywords: 'vision language model, image generation, object detection, multimodal learning' },
  NLP: { label: '自然语言处理', venueGroup: 'NLP', keywords: 'large language model, retrieval augmented generation, language agent, text generation' },
  ML: { label: '机器学习', venueGroup: 'ML', keywords: 'foundation model, representation learning, efficient training, model adaptation' },
  AI: { label: '人工智能', venueGroup: 'AI', keywords: 'multi-agent system, reasoning, planning, embodied intelligence' },
  IRDM: { label: '信息检索/数据挖掘', venueGroup: 'IR/DM', keywords: 'information retrieval, recommender system, knowledge graph, data mining' },
}
const TIME_PRESETS = {
  '1y': { label: '近 1 年', days: 365 },
  '2y': { label: '近 2 年', days: 730 },
  '3y': { label: '近 3 年', days: 1095 },
  '5y': { label: '近 5 年', days: 1825 },
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    let message = payload?.detail || payload || `Request failed: ${response.status}`
    if (typeof message === 'string' && /504 Gateway Time-out|Gateway Time-out|<html/i.test(message)) {
      message = '请求超时：论文源响应较慢，请稍后重试，或暂时切换到 arXiv 预印本。'
    }
    throw new Error(message)
  }
  return payload
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function mergeTrace(existing, incoming = [], label) {
  const stamp = new Date().toLocaleTimeString()
  return [
    ...incoming.map((step) => ({ ...step, group: label, stamp })),
    ...existing,
  ].slice(0, 120)
}

function venueGroupLabel(group) {
  const venues = VENUE_GROUPS[group] || []
  if (group === 'ALL') return '全部会议'
  return `${group}：${venues.join(' / ')}`
}

function timePresetLabel(value) {
  return TIME_PRESETS[value]?.label || TIME_PRESETS['1y'].label
}

function daysFromTimePreset(value) {
  return TIME_PRESETS[value]?.days || TIME_PRESETS['1y'].days
}

function timePresetFromDays(days) {
  const numeric = Number(days || 365)
  return Object.entries(TIME_PRESETS).find(([, preset]) => preset.days === numeric)?.[0] || '1y'
}

function normalizeKeywords(value) {
  return String(value || '').split(/[,，;；\n]+/).map((item) => item.trim()).filter(Boolean)
}

function keywordsToQuery(value) {
  return normalizeKeywords(value).join(', ')
}

function paperKey(paper) {
  return paper?.id || paper?.title
}

function normalizeQueryKey(value) {
  return String(value || '').trim().replace(/\s+/g, ' ').toLowerCase()
}

function normalizeTitle(value) {
  return String(value || '').trim().replace(/\s+/g, ' ').replace(/[^\p{L}\p{N}]+/gu, ' ').toLowerCase()
}

function stableTopicId(query) {
  const normalized = normalizeQueryKey(query) || 'topic'
  let hash = 0
  Array.from(normalized).forEach((char) => {
    hash = ((hash * 31) + char.codePointAt(0)) >>> 0
  })
  return `topic-${hash.toString(36)}`
}

function stableFieldId(fieldName, query) {
  return stableTopicId(fieldName || query)
}

function extractArxivId(paper) {
  const external = paper?.external_ids || paper?.externalIds || {}
  if (external.ArXiv) return String(external.ArXiv).toLowerCase()
  const value = `${paper?.arxiv_url || ''} ${paper?.pdf_url || ''}`
  const match = value.match(/(?:abs|pdf)\/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?/i)
  return match ? match[1].toLowerCase() : ''
}

function paperIdentityKeys(paper) {
  const external = paper?.external_ids || paper?.externalIds || {}
  const keys = []
  if (paper?.id) keys.push(`id:${String(paper.id).toLowerCase()}`)
  if (external.DOI) keys.push(`doi:${String(external.DOI).toLowerCase()}`)
  const arxivId = extractArxivId(paper)
  if (arxivId) keys.push(`arxiv:${arxivId}`)
  const title = normalizeTitle(paper?.title)
  if (title) keys.push(`title:${title}`)
  return keys
}

function topicAnchorId(topic) {
  return `watch-${topic.id}`
}

function preferPaper(current, incoming) {
  const currentSemantic = String(current?.source || '').includes('Semantic Scholar')
  const incomingSemantic = String(incoming?.source || '').includes('Semantic Scholar')
  const primary = incomingSemantic && !currentSemantic ? incoming : current
  const secondary = primary === incoming ? current : incoming
  return {
    ...secondary,
    ...primary,
    pdf_url: primary.pdf_url || secondary.pdf_url || '',
    arxiv_url: primary.arxiv_url || secondary.arxiv_url || '',
    paper_url: primary.paper_url || secondary.paper_url || '',
    summary: primary.summary || secondary.summary || '',
    citation_count: Math.max(primary.citation_count || 0, secondary.citation_count || 0),
    influential_citation_count: Math.max(primary.influential_citation_count || 0, secondary.influential_citation_count || 0),
    is_new: Boolean(current?.is_new || incoming?.is_new),
  }
}

function mergePaperLists(existing = [], incoming = []) {
  const merged = []
  const keyToIndex = new Map()
  const indexPaper = (paper, index) => {
    paperIdentityKeys(paper).forEach((key) => keyToIndex.set(key, index))
  }
  const add = (paper) => {
    const keys = paperIdentityKeys(paper)
    const found = keys.map((key) => keyToIndex.get(key)).find((index) => index !== undefined)
    if (found !== undefined) {
      merged[found] = preferPaper(merged[found], paper)
      indexPaper(merged[found], found)
      return
    }
    merged.push(paper)
    indexPaper(paper, merged.length - 1)
  }
  incoming.forEach(add)
  existing.forEach(add)
  return merged
}

function summarizeSources(papers = []) {
  return papers.reduce((summary, paper) => {
    const source = String(paper.source || '')
    if (source.includes('Semantic Scholar')) summary.semantic += 1
    if (source.startsWith('arXiv')) summary.arxiv += 1
    return summary
  }, { semantic: 0, arxiv: 0 })
}

function normalizeRun(run = {}) {
  return {
    id: run.id || `${Date.now()}`,
    sourceMode: run.sourceMode || run.source_mode || 'conference',
    venueGroup: run.venueGroup || run.venue_group || 'ML',
    venues: run.venues || [],
    sortBy: run.sortBy || run.sort_by || 'recommended',
    timePreset: run.timePreset || run.time_preset || timePresetFromDays(run.days),
    days: Number(run.days || 365),
    maxResults: Number(run.maxResults || run.max_results || 10),
    checkedAt: run.checkedAt || run.checked_at || '',
    statusMessage: run.statusMessage || run.status_message || '',
    sourceWarning: run.sourceWarning || run.source_warning || '',
    sortWarning: run.sortWarning || run.sort_warning || '',
    error: run.error || '',
    totalCount: run.totalCount || run.total_count || 0,
    newCount: run.newCount || run.new_count || 0,
  }
}

function normalizeWatchTopic(topic = {}) {
  const query = String(topic.query || topic.name || '').trim()
  const fieldName = String(topic.fieldName || topic.field_name || topic.name || query || '').trim()
  const timePreset = topic.timePreset || topic.time_preset || timePresetFromDays(topic.days)
  const normalized = {
    ...topic,
    id: topic.id && String(topic.id).startsWith('topic-') ? topic.id : stableFieldId(fieldName, query),
    fieldName,
    query,
    maxResults: Number(topic.maxResults || topic.max_results || 10),
    timePreset,
    days: Number(topic.days || daysFromTimePreset(timePreset)),
    sourceMode: topic.sourceMode || topic.source_mode || 'conference',
    venueGroup: topic.venueGroup || topic.venue_group || 'ML',
    sortBy: topic.sortBy || topic.sort_by || 'recommended',
    venues: topic.venues || [],
    seenIds: topic.seenIds || topic.seen_ids || [],
    baselineDone: Boolean(topic.baselineDone || topic.baseline_done),
    baselineCount: topic.baselineCount || topic.baseline_count || 0,
    papers: topic.papers || [],
    checkedAt: topic.checkedAt || topic.checked_at || '',
    newCount: topic.newCount || topic.new_count || 0,
    totalCount: topic.totalCount || topic.total_count || (topic.papers || []).length,
    error: topic.error || '',
    statusMessage: topic.statusMessage || topic.status_message || '',
    sourceWarning: topic.sourceWarning || topic.source_warning || '',
    sortWarning: topic.sortWarning || topic.sort_warning || '',
    runs: (topic.runs || []).map(normalizeRun),
    brief: topic.brief || '',
    briefUpdatedAt: topic.briefUpdatedAt || '',
  }
  if (!normalized.runs.length && (normalized.checkedAt || normalized.statusMessage || normalized.error)) {
    normalized.runs = [normalizeRun(normalized)]
  }
  normalized.sourceSummary = summarizeSources(normalized.papers)
  return normalized
}

function migrateWatchTopics(items = []) {
  const byQuery = new Map()
  items.map(normalizeWatchTopic).forEach((topic) => {
    const key = normalizeQueryKey(topic.fieldName || topic.query)
    if (!key) return
    const existing = byQuery.get(key)
    if (!existing) {
      byQuery.set(key, topic)
      return
    }
    const papers = mergePaperLists(existing.papers, topic.papers)
    const runs = [...topic.runs, ...existing.runs].slice(0, 20)
    byQuery.set(key, {
      ...existing,
      ...topic,
      id: existing.id,
      papers,
      runs,
      seenIds: Array.from(new Set([...(existing.seenIds || []), ...(topic.seenIds || [])])),
      totalCount: papers.length,
      newCount: papers.filter((paper) => paper.is_new).length,
      sourceSummary: summarizeSources(papers),
    })
  })
  return Array.from(byQuery.values())
}

function loadWatchTopics() {
  const current = loadJson(WATCH_STORAGE_KEY, null)
  if (current) return migrateWatchTopics(current)
  const legacy = loadJson(LEGACY_WATCH_STORAGE_KEY, null)
  if (legacy) return migrateWatchTopics(legacy)
  return migrateWatchTopics(loadJson(OLDER_WATCH_STORAGE_KEY, []))
}

function loadWorkbenchPapers() {
  const current = loadJson(WORKBENCH_STORAGE_KEY, null)
  if (Array.isArray(current)) return current
  if (current?.papers && Array.isArray(current.papers)) return current.papers
  const legacy = loadJson(LEGACY_WORKBENCH_STORAGE_KEY, null)
  if (Array.isArray(legacy)) return legacy
  if (legacy?.papers && Array.isArray(legacy.papers)) return legacy.papers
  return loadJson(OLDER_WORKBENCH_STORAGE_KEY, [])
}

function loadAnalysisCache() {
  return loadJson(ANALYSIS_CACHE_STORAGE_KEY, {})
}

function loadReadingRecords() {
  return loadJson(READING_RECORD_STORAGE_KEY, [])
}

function loadQaHistories() {
  return loadJson(QA_HISTORY_STORAGE_KEY, {})
}

function mergeQaHistory(existing = [], incoming) {
  if (!incoming?.question) return existing
  const key = normalizeQueryKey(incoming.question)
  return [
    { id: incoming.id || `${Date.now()}`, ...incoming },
    ...existing.filter((item) => normalizeQueryKey(item.question) !== key),
  ].slice(0, 10)
}

function App() {
  const [query, setQuery] = useState('large language model agent')
  const [health, setHealth] = useState(null)
  const [papers, setPapers] = useState(loadWorkbenchPapers)
  const [rankedPapers, setRankedPapers] = useState([])
  const [selectedIds, setSelectedIds] = useState([])
  const [graph, setGraph] = useState({ nodes: [], edges: [] })
  const [trends, setTrends] = useState(null)
  const [literatureReview, setLiteratureReview] = useState('')
  const [uploaded, setUploaded] = useState(null)
  const [summary, setSummary] = useState('')
  const [reviewer, setReviewer] = useState('')
  const [visualAssets, setVisualAssets] = useState([])
  const [question, setQuestion] = useState('这篇论文的核心方法是什么？')
  const [qa, setQa] = useState(null)
  const [trace, setTrace] = useState([])
  const [traceExpanded, setTraceExpanded] = useState(false)
  const [loading, setLoading] = useState('')
  const [error, setError] = useState('')
  const [sectionJobs, setSectionJobs] = useState({})
  const [watchTopics, setWatchTopics] = useState(loadWatchTopics)
  const [analysisCache, setAnalysisCache] = useState(loadAnalysisCache)
  const [readingRecords, setReadingRecords] = useState(loadReadingRecords)
  const [selectedReadingRecords, setSelectedReadingRecords] = useState([])
  const [qaHistories, setQaHistories] = useState(loadQaHistories)
  const [selectedQaHistory, setSelectedQaHistory] = useState([])
  const [expandedTopics, setExpandedTopics] = useState({})
  const [expandedRuns, setExpandedRuns] = useState({})
  const [drawerTab, setDrawerTab] = useState('list')
  const [analysisFieldFilter, setAnalysisFieldFilter] = useState('ALL')
  const [drawerPulse, setDrawerPulse] = useState(false)
  const [viewMode, setViewMode] = useState(() => (window.location.hash === '#workbench' ? 'workbench' : 'inbox'))
  const [addedPaperIds, setAddedPaperIds] = useState({})
  const [watchDraft, setWatchDraft] = useState({
    fieldPreset: 'NLP',
    fieldName: FIELD_PRESETS.NLP.label,
    query: FIELD_PRESETS.NLP.keywords,
    maxResults: 10,
    timePreset: '1y',
    days: 365,
    sourceMode: 'preprint',
    venueGroup: FIELD_PRESETS.NLP.venueGroup,
    sortBy: 'recommended',
  })
  const [refreshingWatch, setRefreshingWatch] = useState('')
  const [refreshingTopics, setRefreshingTopics] = useState({})
  const [now, setNow] = useState(Date.now())
  const [pdfJobs, setPdfJobs] = useState({})
  const [llmJobs, setLlmJobs] = useState({})
  const markdownMode = 'rendered'
  const autoRefreshRef = useRef(false)
  const sourceInitializedRef = useRef(false)
  const uploadInputRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    async function checkHealth(attempt = 1) {
      try {
        const payload = await api('/api/health')
        if (!cancelled) {
          setHealth(payload)
          if (!sourceInitializedRef.current) {
            sourceInitializedRef.current = true
            setWatchDraft((draft) => ({
              ...draft,
              sourceMode: payload.semantic_scholar_configured ? 'conference' : 'preprint',
            }))
          }
          setError('')
        }
      } catch (err) {
        if (cancelled) return
        if (attempt < 8) {
          setTimeout(() => checkHealth(attempt + 1), 1000 * attempt)
        } else {
          setError(err.message)
        }
      }
    }
    checkHealth()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    const handleHash = () => setViewMode(window.location.hash === '#workbench' ? 'workbench' : 'inbox')
    window.addEventListener('hashchange', handleHash)
    window.addEventListener('popstate', handleHash)
    return () => {
      window.removeEventListener('hashchange', handleHash)
      window.removeEventListener('popstate', handleHash)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(WATCH_STORAGE_KEY, JSON.stringify(watchTopics))
  }, [watchTopics])

  useEffect(() => {
    localStorage.setItem(WORKBENCH_STORAGE_KEY, JSON.stringify({ papers }))
  }, [papers])

  useEffect(() => {
    localStorage.setItem(ANALYSIS_CACHE_STORAGE_KEY, JSON.stringify(analysisCache))
  }, [analysisCache])

  useEffect(() => {
    localStorage.setItem(READING_RECORD_STORAGE_KEY, JSON.stringify(readingRecords.slice(0, 30)))
  }, [readingRecords])

  useEffect(() => {
    localStorage.setItem(QA_HISTORY_STORAGE_KEY, JSON.stringify(qaHistories))
  }, [qaHistories])

  useEffect(() => {
    if (autoRefreshRef.current || !watchTopics.length) return
    autoRefreshRef.current = true
    const autoTopics = watchTopics.filter((topic) => normalizeWatchTopic(topic).sourceMode !== 'conference')
    if (autoTopics.length) {
      refreshWatch(autoTopics, '自动刷新论文收件箱')
    }
  }, [])

  useEffect(() => {
    if (!uploaded) return
    const cachedSummary = analysisCache[cacheKey('summary')]?.value
    const cachedReviewer = analysisCache[cacheKey('reviewer')]?.value
    if (cachedSummary && !summary) {
      if (typeof cachedSummary === 'string') {
        setSummary(cachedSummary)
      } else {
        setSummary(cachedSummary.content || '')
        setVisualAssets(cachedSummary.visualAssets || cachedSummary.visual_assets || [])
      }
    }
    if (cachedReviewer && !reviewer) setReviewer(cachedReviewer)
    setSelectedQaHistory([])
  }, [uploaded?.paper_id])

  const visiblePapers = rankedPapers.length ? rankedPapers : papers
  const selectedPapers = useMemo(
    () => visiblePapers.filter((paper) => selectedIds.includes(paper.id)),
    [visiblePapers, selectedIds],
  )
  const workbenchIds = useMemo(() => new Set(papers.map(paperKey)), [papers])
  const baseAnalysisPapers = selectedPapers.length ? selectedPapers : visiblePapers
  const workbenchFields = useMemo(() => {
    const fields = Array.from(new Set(visiblePapers.map((paper) => paper.sourceField || paper.source_field || '').filter(Boolean)))
    return fields
  }, [visiblePapers])
  const analysisPapers = analysisFieldFilter === 'ALL'
    ? baseAnalysisPapers
    : baseAnalysisPapers.filter((paper) => (paper.sourceField || paper.source_field || '') === analysisFieldFilter)
  const analysisFields = useMemo(() => {
    const fields = Array.from(new Set(analysisPapers.map((paper) => paper.sourceField || paper.source_field || '').filter(Boolean)))
    return fields
  }, [analysisPapers])
  const qaHistory = useMemo(() => {
    if (!uploaded) return []
    return (qaHistories[uploaded.paper_id] || []).map((item, index) => ({
      ...item,
      id: item.id || `${normalizeQueryKey(item.question)}-${index}`,
    }))
      .sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')))
  }, [qaHistories, uploaded?.paper_id])

  useEffect(() => {
    if (!uploaded) return
    setReadingRecords((records) => {
      const existing = records.find((record) => record.paper_id === uploaded.paper_id) || {}
      const paperQaHistory = qaHistories[uploaded.paper_id] || []
      const nextRecord = {
        ...existing,
        ...uploaded,
        paper_id: uploaded.paper_id,
        filename: uploaded.filename,
        sourcePaper: uploaded.sourcePaper || existing.sourcePaper || null,
        updatedAt: new Date().toISOString(),
        summary: summary || existing.summary || '',
        reviewer: reviewer || existing.reviewer || '',
        visualAssets: visualAssets.length ? visualAssets : (existing.visualAssets || []),
        qaCount: paperQaHistory.length || existing.qaCount || 0,
      }
      return [nextRecord, ...records.filter((record) => record.paper_id !== uploaded.paper_id)].slice(0, 30)
    })
  }, [uploaded?.paper_id, summary, reviewer, visualAssets, qaHistories])
  const trackedPaperCount = watchTopics.reduce((sum, topic) => sum + (topic.papers?.length || 0), 0)
  const trackedNewCount = watchTopics.reduce((sum, topic) => sum + (topic.newCount || 0), 0)
  const visibleTrace = traceExpanded ? trace : trace.slice(0, 8)

  async function runAction(label, fn, jobKey = '') {
    setLoading(label)
    setError('')
    if (jobKey) {
      setSectionJobs((jobs) => ({
        ...jobs,
        [jobKey]: { status: 'running', label, startedAt: Date.now(), error: '' },
      }))
    }
    try {
      await fn()
      if (jobKey) {
        setSectionJobs((jobs) => ({
          ...jobs,
          [jobKey]: { ...jobs[jobKey], status: 'completed', label, completedAt: Date.now(), error: '' },
        }))
      }
    } catch (err) {
      setError(err.message)
      if (jobKey) {
        setSectionJobs((jobs) => ({
          ...jobs,
          [jobKey]: { ...jobs[jobKey], status: 'failed', label, completedAt: Date.now(), error: err.message },
        }))
      }
    } finally {
      setLoading('')
    }
  }

  function updateTopicDraft(field, value) {
    setWatchDraft((draft) => {
      if (field === 'fieldPreset') {
        const preset = FIELD_PRESETS[value] || FIELD_PRESETS.CUSTOM
        return {
          ...draft,
          fieldPreset: value,
          fieldName: value === 'CUSTOM' ? draft.fieldName : preset.label,
          query: value === 'CUSTOM' ? draft.query : preset.keywords,
          venueGroup: preset.venueGroup,
        }
      }
      if (field === 'timePreset') {
        return { ...draft, timePreset: value, days: daysFromTimePreset(value) }
      }
      return { ...draft, [field]: value }
    })
  }

  function toggleTopicExpanded(id) {
    setExpandedTopics((items) => ({ ...items, [id]: !items[id] }))
  }

  function toggleRunsExpanded(id) {
    setExpandedRuns((items) => ({ ...items, [id]: !items[id] }))
  }

  function openWorkbench(tab = drawerTab) {
    setDrawerTab(tab)
    setViewMode('workbench')
    if (window.location.hash !== '#workbench') {
      window.history.pushState(null, '', '#workbench')
    }
  }

  function returnToInbox() {
    setViewMode('inbox')
    if (window.location.hash !== '#inbox') {
      window.history.pushState(null, '', '#inbox')
    }
  }

  function openManualUpload() {
    openWorkbench('reading')
    uploadInputRef.current?.click()
  }

  function buildWatchRequest(topics) {
    return topics.map((topic) => ({
      id: topic.id,
      name: topic.fieldName || topic.query,
      query: topic.query,
      max_results: topic.maxResults,
      days: topic.days,
      seen_ids: topic.seenIds,
      source_mode: topic.sourceMode,
      venue_group: topic.venueGroup,
      sort_by: topic.sortBy,
      venues: [],
      baseline_done: topic.baselineDone,
    }))
  }

  async function refreshWatch(topicsToRefresh = watchTopics, label = '刷新论文收件箱') {
    const normalized = topicsToRefresh.map(normalizeWatchTopic)
    if (!normalized.length) return
    setRefreshingWatch(label)
    const startedAt = Date.now()
    setRefreshingTopics((items) => ({
      ...items,
      ...Object.fromEntries(normalized.map((topic) => [topic.id, { label, startedAt }])),
    }))
    setError('')
    try {
      const payload = await api('/api/watch/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topics: buildWatchRequest(normalized) }),
      })
      const results = new Map(payload.topics.map((topic) => [topic.id, topic]))
      setWatchTopics((items) => {
        const byId = new Map(items.map((item) => {
          const normalizedItem = normalizeWatchTopic(item)
          return [normalizedItem.id, normalizedItem]
        }))
        normalized.forEach((topic) => {
          byId.set(topic.id, mergeWatchTopic(byId.get(topic.id) || topic, results.get(topic.id)))
        })
        return Array.from(byId.values())
      })
      setTrace((items) => mergeTrace(items, payload.trace, '论文追踪收件箱'))
    } catch (err) {
      setError(err.message)
    } finally {
      setRefreshingWatch('')
      setRefreshingTopics((items) => {
        const next = { ...items }
        normalized.forEach((topic) => {
          delete next[topic.id]
        })
        return next
      })
    }
  }

  function mergeWatchTopic(topic, result) {
    const base = normalizeWatchTopic(topic)
    if (!result) return base
    const incomingPapers = result.papers || []
    const mergedPapers = mergePaperLists(base.papers, incomingPapers)
    const paperIds = mergedPapers.flatMap((paper) => [paper.id, ...paperIdentityKeys(paper)]).filter(Boolean)
    const seenIds = Array.from(new Set([...(topic.seenIds || []), ...paperIds]))
    const statusMessage = result.source_warning && incomingPapers.length
      ? '部分补充完成'
      : result.status_message || ''
    const run = normalizeRun({
      id: `${Date.now()}-${result.source_mode}`,
      sourceMode: result.source_mode,
      venueGroup: result.venue_group,
      venues: result.venues || [],
      sortBy: result.sort_by || base.sortBy || 'recommended',
      timePreset: base.timePreset,
      days: base.days,
      maxResults: base.maxResults,
      checkedAt: result.checked_at,
      statusMessage,
      sourceWarning: result.source_warning || '',
      sortWarning: result.sort_warning || '',
      error: result.error || '',
      totalCount: result.total_count || incomingPapers.length,
      newCount: result.new_count || 0,
    })
    return {
      ...base,
      sourceMode: result.source_mode,
      venueGroup: result.venue_group,
      sortBy: result.sort_by || topic.sortBy || 'recommended',
      venues: result.venues || topic.venues || [],
      checkedAt: result.checked_at,
      papers: mergedPapers,
      newCount: result.new_count,
      totalCount: mergedPapers.length,
      baselineCount: result.baseline_count || topic.baselineCount || 0,
      baselineDone: true,
      error: result.error || '',
      statusMessage,
      sourceWarning: result.source_warning || '',
      sortWarning: result.sort_warning || '',
      runs: [run, ...(base.runs || [])].slice(0, 20),
      sourceSummary: summarizeSources(mergedPapers),
      seenIds,
    }
  }

  async function addWatchTopic() {
    const topicQuery = keywordsToQuery(watchDraft.query)
    const fieldName = (watchDraft.fieldName || topicQuery).trim()
    if (!topicQuery) {
      setError('研究关键词不能为空。')
      return
    }
    if (!fieldName) {
      setError('关注领域不能为空。')
      return
    }
    setQuery(topicQuery)
    const topicId = stableFieldId(fieldName, topicQuery)
    const topic = normalizeWatchTopic({
      id: topicId,
      fieldName,
      query: topicQuery,
      maxResults: Number(watchDraft.maxResults),
      timePreset: watchDraft.timePreset,
      days: daysFromTimePreset(watchDraft.timePreset),
      sourceMode: watchDraft.sourceMode,
      venueGroup: watchDraft.venueGroup,
      sortBy: watchDraft.sortBy,
      venues: [],
    })
    const existing = watchTopics.map(normalizeWatchTopic).find((item) => normalizeQueryKey(item.fieldName || item.query) === normalizeQueryKey(fieldName))
    const nextTopic = existing ? { ...existing, ...topic, id: existing.id, papers: existing.papers, runs: existing.runs, seenIds: existing.seenIds, baselineDone: existing.baselineDone } : topic
    setWatchTopics((items) => existing ? items.map((item) => normalizeWatchTopic(item).id === existing.id ? nextTopic : item) : [nextTopic, ...items])
    await refreshWatch([nextTopic], existing ? '更新该领域' : '获取该领域')
  }

  function removeWatchTopic(id) {
    setWatchTopics((items) => items.filter((topic) => topic.id !== id))
  }

  function clearSeenIds(topic) {
    setWatchTopics((items) => items.map((item) => item.id === topic.id ? { ...item, seenIds: [], newCount: 0, papers: (item.papers || []).map((paper) => ({ ...paper, is_new: false })) } : item))
  }

  function withSourceField(paper, topic = null) {
    if (!topic) return paper
    const normalized = normalizeWatchTopic(topic)
    return {
      ...paper,
      sourceField: normalized.fieldName || normalized.query,
      sourceQuery: normalized.query,
    }
  }

  function findPaperById(id) {
    return [
      ...papers,
      ...watchTopics.flatMap((topic) => (topic.papers || [])),
    ].find((paper) => paper.id === id)
  }

  function addToWorkbench(incoming, topic = null) {
    const nextItems = Array.isArray(incoming) ? incoming : [incoming]
    const enrichedItems = nextItems.map((paper) => withSourceField(paper, topic))
    setPapers((items) => {
      const merged = mergePaperLists(items, enrichedItems.map((paper) => ({ ...paper, is_new: false })))
      setSelectedIds((ids) => Array.from(new Set([...ids, ...enrichedItems.map((paper) => paper.id)])).slice(0, 30))
      return merged
    })
    setRankedPapers([])
    setDrawerPulse(true)
    setAddedPaperIds((items) => ({
      ...items,
      ...Object.fromEntries(enrichedItems.map((paper) => [paper.id, true])),
    }))
    window.setTimeout(() => setDrawerPulse(false), 900)
    window.setTimeout(() => {
      setAddedPaperIds((items) => {
        const next = { ...items }
        enrichedItems.forEach((paper) => {
          delete next[paper.id]
        })
        return next
      })
    }, 1400)
  }

  function useTopicForAnalysis(topic) {
    const topicPapers = (topic.papers || []).map((paper) => withSourceField({ ...paper, is_new: false }, topic))
    setQuery(topic.query)
    setPapers((items) => mergePaperLists(items, topicPapers))
    setRankedPapers([])
    setSelectedIds((ids) => Array.from(new Set([...ids, ...topicPapers.slice(0, 8).map((paper) => paper.id)])))
    openWorkbench('list')
  }

  function scrollToTopic(topic) {
    if (viewMode !== 'inbox') {
      returnToInbox()
      window.setTimeout(() => {
        document.getElementById(topicAnchorId(topic))?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 80)
      return
    }
    document.getElementById(topicAnchorId(topic))?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  function clearWorkbench() {
    setPapers([])
    setRankedPapers([])
    setSelectedIds([])
    setGraph({ nodes: [], edges: [] })
    setTrends(null)
    setLiteratureReview('')
  }

  function openReadingRecord(record) {
    setUploaded({
      paper_id: record.paper_id,
      filename: record.filename,
      page_count: record.page_count,
      chunk_count: record.chunk_count,
      char_count: record.char_count,
      trace: record.trace || [],
      sourcePaper: record.sourcePaper || null,
    })
    setSummary(record.summary || '')
    setReviewer(record.reviewer || '')
    setVisualAssets(record.visualAssets || [])
    const latestQa = (qaHistories[record.paper_id] || [])[0]
    setQa(latestQa?.payload || null)
    if (latestQa?.question) setQuestion(latestQa.question)
    openWorkbench('reading')
    setTrace((items) => mergeTrace(items, [{ name: '打开精读历史', status: 'completed', detail: record.filename, elapsed_ms: 0 }], '本地缓存'))
  }

  function toggleReadingRecord(id) {
    setSelectedReadingRecords((ids) => ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id])
  }

  function deleteSelectedReadingRecords() {
    const selected = new Set(selectedReadingRecords)
    setReadingRecords((records) => records.filter((record) => !selected.has(record.paper_id)))
    if (uploaded && selected.has(uploaded.paper_id)) {
      setUploaded(null)
      setSummary('')
      setReviewer('')
      setQa(null)
    }
    setSelectedReadingRecords([])
  }

  function clearReadingRecords() {
    setReadingRecords([])
    setSelectedReadingRecords([])
  }

  function removeFromWorkbench(paperId) {
    setPapers((items) => items.filter((paper) => paper.id !== paperId))
    setRankedPapers((items) => items.filter((paper) => paper.id !== paperId))
    setSelectedIds((ids) => ids.filter((id) => id !== paperId))
  }

  function cacheKey(kind, extra = '') {
    const model = health?.llm_model || 'default-model'
    if (kind === 'summary') {
      return `${kind}:visual-assets-v2:${model}:${uploaded?.paper_id || 'none'}:${uploaded?.chunk_count || 0}:${extra}`
    }
    if (kind === 'reviewer') {
      return `${kind}:${model}:${uploaded?.paper_id || 'none'}:${uploaded?.chunk_count || 0}:${extra}`
    }
    if (kind === 'qa') {
      return `${kind}:visual-v2:${model}:${uploaded?.paper_id || 'none'}:${uploaded?.chunk_count || 0}:${normalizeQueryKey(question)}`
    }
    const ids = analysisPapers.map((paper) => paper.id).sort().join('|')
    return `${kind}:${model}:${normalizeQueryKey(query)}:${ids}:${extra}`
  }

  function saveCache(key, value) {
    setAnalysisCache((cache) => ({
      ...cache,
      [key]: { value, updatedAt: new Date().toISOString() },
    }))
    setTrace((items) => mergeTrace(items, [{ name: '写入本地分析缓存', status: 'completed', detail: key, elapsed_ms: 0 }], '本地缓存'))
  }

  function loadCache(key) {
    const cached = analysisCache[key]
    if (cached) {
      setTrace((items) => mergeTrace(items, [{ name: '命中本地分析缓存', status: 'completed', detail: key, elapsed_ms: 0 }], '本地缓存'))
    }
    return cached?.value
  }

  async function handleRank() {
    await runAction('排序工作台论文', async () => {
      const payload = await api('/api/rank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, papers }),
      })
      setRankedPapers(payload.papers)
      setSelectedIds(payload.papers.slice(0, 8).map((paper) => paper.id))
      setTrace((items) => mergeTrace(items, payload.trace, '论文排序'))
    }, 'ranking')
  }

  async function handleTrends() {
    await runAction('趋势分析', async () => {
      const key = cacheKey('trends')
      const cached = loadCache(key)
      if (cached) {
        setTrends(cached)
        return
      }
      const source = analysisPapers
      const analysisQuery = analysisFields.length > 1
        ? `跨领域分析：${analysisFields.join('、')}。请按领域分组比较研究趋势。`
        : query
      const payload = await api('/api/trends', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: analysisQuery, papers: source }),
      })
      setTrends(payload)
      saveCache(key, payload)
      setTrace((items) => mergeTrace(items, payload.trace, '趋势分析'))
    }, 'trends')
  }

  async function handleGraph() {
    await runAction('生成图谱', async () => {
      const key = cacheKey('graph')
      const cached = loadCache(key)
      if (cached) {
        setGraph(cached)
        return
      }
      const payload = await api('/api/graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ papers: analysisPapers, similarity_threshold: 0.38 }),
      })
      setGraph(payload)
      saveCache(key, payload)
      setTrace((items) => mergeTrace(items, payload.trace, '论文图谱'))
    }, 'graph')
  }

  async function handleLiteratureReview() {
    await runAction('生成综述', async () => {
      const source = analysisPapers.slice(0, 16)
      const key = cacheKey('literature', source.map((paper) => paper.id).sort().join('|'))
      const cached = loadCache(key)
      if (cached) {
        setLiteratureReview(cached)
        return
      }
      const analysisQuery = analysisFields.length > 1
        ? `跨领域文献综述：${analysisFields.join('、')}。请先按领域分组，再比较共同主题、方法差异和可迁移启发。`
        : query
      const payload = await api('/api/literature-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: analysisQuery, papers: source, graph }),
      })
      setLiteratureReview(payload.content)
      saveCache(key, payload.content)
      setTrace((items) => mergeTrace(items, payload.trace, '文献综述'))
    }, 'literature')
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    openWorkbench('reading')
    await runAction('上传 PDF', async () => {
      const form = new FormData()
      form.append('file', file)
      const payload = await api('/api/papers/upload', { method: 'POST', body: form })
      setUploaded({ ...payload, sourcePaper: null })
      setSummary('')
      setReviewer('')
      setVisualAssets([])
      setQa(null)
      setTrace((items) => mergeTrace(items, payload.trace, 'PDF 精读'))
    })
  }

  async function handleIngestPdf(paper) {
    if (!paper.pdf_url) {
      setError('这篇论文没有可直接解析的开放 PDF 链接。')
      return
    }
    setError('')
    setPdfJobs((jobs) => ({
      ...jobs,
      [paper.id]: { status: 'queued', stage: '等待中', error: '', trace: [], startedAt: Date.now(), sourcePaper: paper },
    }))
    try {
      const created = await api('/api/papers/from-url/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdf_url: paper.pdf_url, title: paper.title }),
      })
      setPdfJobs((jobs) => ({
        ...jobs,
        [paper.id]: { ...jobs[paper.id], ...created, startedAt: Date.now() },
      }))
      pollPdfJob(paper.id, created.job_id)
    } catch (err) {
      setPdfJobs((jobs) => ({
        ...jobs,
        [paper.id]: { ...jobs[paper.id], status: 'failed', stage: '失败', error: err.message },
      }))
    }
  }

  async function pollPdfJob(paperId, jobId) {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      await sleep(1200)
      try {
        const payload = await api(`/api/jobs/${jobId}`)
        setPdfJobs((jobs) => ({
          ...jobs,
          [paperId]: {
            ...jobs[paperId],
            ...payload,
            elapsedSeconds: Math.round((Date.now() - (jobs[paperId]?.startedAt || Date.now())) / 1000),
          },
        }))
        if (payload.status === 'completed') {
          const sourcePaper = findPaperById(paperId) || null
          setUploaded({ ...payload.result, sourcePaper })
          setSummary('')
          setReviewer('')
          setVisualAssets([])
          setQa(null)
          setTrace((items) => mergeTrace(items, payload.trace, '在线 PDF 精读'))
          openWorkbench('reading')
          return
        }
        if (payload.status === 'failed') {
          setTrace((items) => mergeTrace(items, payload.trace, '在线 PDF 精读'))
          return
        }
      } catch (err) {
        setPdfJobs((jobs) => ({
          ...jobs,
          [paperId]: { ...jobs[paperId], status: 'failed', stage: '失败', error: err.message },
        }))
        return
      }
    }
  }

  async function runLlmJob(kind, force = false) {
    if (!uploaded) return
    const label = kind === 'summary' ? '结构化总结' : 'Reviewer 分析'
    const setter = kind === 'summary' ? setSummary : setReviewer
    const key = cacheKey(kind)
    if (!force) {
      const cached = loadCache(key)
      if (cached) {
        if (kind === 'summary' && typeof cached !== 'string') {
          setter(cached.content || '')
          setVisualAssets(cached.visualAssets || cached.visual_assets || [])
        } else {
          setter(cached)
        }
        setLlmJobs((jobs) => ({
          ...jobs,
          [kind]: { status: 'completed', stage: '已从历史记录加载', error: '', elapsedSeconds: 0, startedAt: Date.now() },
        }))
        return
      }
    }
    setter('')
    setLlmJobs((jobs) => ({
      ...jobs,
      [kind]: { status: 'queued', stage: '等待中', error: '', elapsedSeconds: 0, startedAt: Date.now() },
    }))
    try {
      const created = await api(`/api/papers/${uploaded.paper_id}/${kind}/jobs`, { method: 'POST' })
      setLlmJobs((jobs) => ({
        ...jobs,
        [kind]: { ...jobs[kind], ...created, startedAt: Date.now() },
      }))
      const events = new EventSource(`${API_BASE}/api/llm-jobs/${created.job_id}/stream`)
      events.onmessage = (event) => {
        const payload = JSON.parse(event.data)
        setLlmJobs((jobs) => ({
          ...jobs,
          [kind]: {
            ...jobs[kind],
            ...payload,
            elapsedSeconds: Math.round((Date.now() - (jobs[kind]?.startedAt || Date.now())) / 1000),
          },
        }))
        if (payload.delta) {
          setter((text) => text + payload.delta)
        }
        if (payload.status === 'completed') {
          if (payload.content) {
            setter(payload.content)
            if (kind === 'summary') {
              const assets = payload.result?.visual_assets || []
              setVisualAssets(assets)
              saveCache(key, { content: payload.content, visualAssets: assets })
            } else {
              saveCache(key, payload.content)
            }
          }
          setTrace((items) => mergeTrace(items, payload.trace, label))
          events.close()
        }
        if (payload.status === 'failed') {
          setError(payload.error || `${label}失败`)
          setTrace((items) => mergeTrace(items, payload.trace, label))
          events.close()
        }
      }
      events.onerror = () => {
        setLlmJobs((jobs) => ({
          ...jobs,
          [kind]: { ...jobs[kind], status: 'failed', stage: '连接中断', error: `${label}流式连接中断` },
        }))
        events.close()
      }
    } catch (err) {
      setLlmJobs((jobs) => ({
        ...jobs,
        [kind]: { ...jobs[kind], status: 'failed', stage: '失败', error: err.message },
      }))
    }
  }

  async function handleQa() {
    if (!uploaded) return
    await runAction('论文问答', async () => {
      const normalizedQuestion = normalizeQueryKey(question)
      const existing = (qaHistories[uploaded.paper_id] || []).find((item) => normalizeQueryKey(item.question) === normalizedQuestion)
      if (existing) {
        setQa(existing.payload)
        return
      }
      const payload = await api(`/api/papers/${uploaded.paper_id}/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 4 }),
      })
      const cachedPayload = { ...payload, question }
      setQa(cachedPayload)
      setQaHistories((items) => ({
        ...items,
        [uploaded.paper_id]: mergeQaHistory(items[uploaded.paper_id] || [], {
          id: `${Date.now()}`,
          question,
          payload: cachedPayload,
          updatedAt: new Date().toISOString(),
        }),
      }))
      setTrace((items) => mergeTrace(items, payload.trace, '论文问答'))
    }, 'qa')
  }

  function togglePaper(id) {
    setSelectedIds((ids) => ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id])
  }

  function toggleQaHistory(id) {
    setSelectedQaHistory((ids) => ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id])
  }

  function deleteSelectedQaHistory() {
    if (!uploaded || !selectedQaHistory.length) return
    const selected = new Set(selectedQaHistory)
    setQaHistories((items) => ({
      ...items,
      [uploaded.paper_id]: (items[uploaded.paper_id] || []).filter((item, index) => !selected.has(item.id || `${normalizeQueryKey(item.question)}-${index}`)),
    }))
    setSelectedQaHistory([])
  }

  function clearCurrentQaHistory() {
    if (!uploaded) return
    setQaHistories((items) => ({ ...items, [uploaded.paper_id]: [] }))
    setSelectedQaHistory([])
    setQa(null)
  }

  const summaryRunning = llmJobs.summary && !['completed', 'failed'].includes(llmJobs.summary.status)
  const reviewerRunning = llmJobs.reviewer && !['completed', 'failed'].includes(llmJobs.reviewer.status)
  const selectedVenues = VENUE_GROUPS[watchDraft.venueGroup] || []
  const sortText = sortLabel(watchDraft.sortBy)
  const timeText = timePresetLabel(watchDraft.timePreset)
  const watchScopeText = watchDraft.sourceMode === 'preprint'
    ? `将从 arXiv 获取${timeText}内与“${watchDraft.fieldName || '关注领域'}”相关的预印本。`
    : watchDraft.sourceMode === 'hybrid'
      ? `将先查 Semantic Scholar 的 ${selectedVenues.join(', ')} 论文；若顶会源限流或超时，会用 arXiv 补充并标为“arXiv 补充”。`
      : `将只查 Semantic Scholar 中 ${selectedVenues.join(', ')} 的${timeText}会议论文；遇到 1 request/s 限流时请稍后重试，不会自动混入 arXiv。`

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <p className="eyebrow">AI Paper Assistant</p>
          <h1>论文阅读助手</h1>
          <p className="subtle">追踪论文，送入工作台分析。</p>
        </div>

        <div className="guide-panel">
          <strong>使用流程</strong>
          <ol>
            <li>选择关注领域并获取论文</li>
            <li>把候选论文加入工作台</li>
            <li>在工作台完成排序、精读和综述</li>
          </ol>
        </div>

        <div className="focus-panel">
          <div className="panel-title">
            <strong>关注领域</strong>
            <InfoTip text="这里显示用户正在长期追踪的研究领域。每个领域可以包含一组关键词，右侧收件箱会把多次获取到的论文自动合并去重。" />
          </div>
          <div className="focus-list">
            {watchTopics.map((rawTopic) => {
              const topic = normalizeWatchTopic(rawTopic)
              const latestRun = topic.runs?.[0]
              return (
                <button className="focus-item" key={topic.id} onClick={() => scrollToTopic(topic)}>
                  <span>{topic.fieldName || topic.query}</span>
                  <small>{topic.totalCount || topic.papers?.length || 0} 篇 · {topic.newCount || 0} new · {latestRun ? sourceLabel(latestRun.sourceMode) : '尚未获取'}</small>
                </button>
              )
            })}
            {!watchTopics.length && <span className="focus-empty">右侧获取论文后会显示关注领域。</span>}
          </div>
        </div>
        <button onClick={() => openWorkbench('list')}>打开工作台</button>

        <div className="health">
          <strong>服务状态</strong>
          <span>{health?.status || 'checking'}</span>
          <span>Embedding: {health?.embedding_model_loaded ? 'ready' : 'loading'}</span>
          <span>LLM: {health?.llm_configured ? 'Kimi/OpenAI API configured' : 'missing key'}</span>
          <span>Semantic Scholar: {health?.semantic_scholar_configured ? 'key configured' : 'missing key'}</span>
          <span>Network: 建议连接 VPN 后使用</span>
        </div>
      </aside>

      <main className={viewMode === 'workbench' ? 'workbench-main' : ''}>
        <header className="topbar">
          <div>
            <p className="eyebrow">{viewMode === 'workbench' ? 'Reading Workspace' : 'Paper Tracking'}</p>
            <h2>{viewMode === 'workbench' ? '工作台' : '论文追踪收件箱'}</h2>
          </div>
          <div className="topbar-actions">
            <div className="status-pill">{refreshingWatch || loading || '就绪'}</div>
          </div>
        </header>

        {error && <div className="error">{error}</div>}

        {viewMode === 'inbox' ? (
          <>
            <section className="section watch-section">
              <div className="section-title">
                <div className="title-with-tip">
                    <h3>论文追踪收件箱</h3>
                  <InfoTip text="顶会论文依赖 Semantic Scholar API Key，免费额度约 1 request/s；如遇限流可稍后重试，或切换 arXiv 预印本。" />
                </div>
                <div className="watch-summary">
                  <span>{watchTopics.length} 领域</span>
                  <span>{trackedPaperCount} papers</span>
                  <span>{trackedNewCount} new</span>
                </div>
              </div>

              <div className="watch-form">
                <label className="field-preset">
                  <span className="label-title">关注领域</span>
                  <select value={watchDraft.fieldPreset} onChange={(event) => updateTopicDraft('fieldPreset', event.target.value)}>
                    {Object.entries(FIELD_PRESETS).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}
                  </select>
                </label>
                {watchDraft.fieldPreset === 'CUSTOM' && (
                  <label className="field-name">
                    <span className="label-title">领域名称</span>
                    <input value={watchDraft.fieldName} onChange={(event) => updateTopicDraft('fieldName', event.target.value)} placeholder="例如：具身智能" />
                  </label>
                )}
                <label className="field-query">
                  <span className="label-title">关键词组</span>
                  <input value={watchDraft.query} onChange={(event) => updateTopicDraft('query', event.target.value)} />
                </label>
                <label className="field-source">
                  <span className="label-title">论文来源</span>
                  <select value={watchDraft.sourceMode} onChange={(event) => updateTopicDraft('sourceMode', event.target.value)}>
                    <option value="conference">顶会论文</option>
                    <option value="hybrid">顶会 + arXiv</option>
                    <option value="preprint">arXiv 预印本</option>
                  </select>
                </label>
                {watchDraft.sourceMode !== 'preprint' && (
                  <label className="field-venue">
                    <span className="label-title">会议范围</span>
                    <select value={watchDraft.venueGroup} onChange={(event) => updateTopicDraft('venueGroup', event.target.value)}>
                      {Object.keys(VENUE_GROUPS).map((group) => <option key={group} value={group}>{venueGroupLabel(group)}</option>)}
                    </select>
                  </label>
                )}
                <label className="field-count">
                  <span className="label-title">返回数量</span>
                  <input type="number" min="1" max="50" value={watchDraft.maxResults} onChange={(event) => updateTopicDraft('maxResults', event.target.value)} />
                </label>
                <label className="field-sort">
                  <span className="label-title">排序方式</span>
                  <select value={watchDraft.sortBy} onChange={(event) => updateTopicDraft('sortBy', event.target.value)}>
                    <option value="recommended">综合推荐</option>
                    <option value="latest">最新优先</option>
                    <option value="citations">引用量优先</option>
                  </select>
                </label>
                <label className="field-days">
                  <span className="label-title">时间范围</span>
                  <select value={watchDraft.timePreset} onChange={(event) => updateTopicDraft('timePreset', event.target.value)}>
                    {Object.entries(TIME_PRESETS).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}
                  </select>
                </label>
                <div className="watch-form-actions">
                  <button onClick={addWatchTopic} disabled={Boolean(refreshingWatch)}>{refreshingWatch ? '刷新中' : '获取并更新该领域'}</button>
                  <button onClick={() => refreshWatch(watchTopics, '更新全部关注领域')} disabled={!watchTopics.length || Boolean(refreshingWatch)} title="按每个关键词最近一次的来源、会议和时间范围重新获取最新论文。">{refreshingWatch ? '刷新中' : '更新全部关注领域'}</button>
                </div>
                <div className="watch-help">
                  <strong>当前范围</strong>
                  <span>{watchScopeText} 排序方式：{sortText}。</span>
                  {refreshingWatch && <span>正在获取：{refreshingWatch}</span>}
                </div>
              </div>

              <div className="watch-list">
                {watchTopics.map((rawTopic) => {
                  const topic = normalizeWatchTopic(rawTopic)
                  const topicRefresh = refreshingTopics[topic.id]
                  const topicElapsed = topicRefresh ? Math.round((now - topicRefresh.startedAt) / 1000) : 0
                  const summary = topic.sourceSummary || summarizeSources(topic.papers)
                  const latestRun = topic.runs?.[0]
                  const topicStatus = topic.sourceWarning && topic.papers?.length ? '部分补充完成' : topic.statusMessage
                  const topicExpanded = Boolean(expandedTopics[topic.id])
                  const topicPapers = topic.papers || []
                  const visibleTopicPapers = topicExpanded ? topicPapers : topicPapers.slice(0, 8)
                  const runsExpanded = Boolean(expandedRuns[topic.id])
                  const visibleRuns = runsExpanded ? (topic.runs || []) : (topic.runs || []).slice(0, 8)
                  return (
                  <article className={`watch-topic ${topicRefresh ? 'refreshing' : ''}`} key={topic.id} id={topicAnchorId(topic)}>
                    <div className="watch-topic-head">
                      <div>
                    <h4>{topic.fieldName || topic.query}</h4>
                        <p>关键词：{topic.query}</p>
                      </div>
                    <div className="watch-actions">
                        <span className="new-pill">{topic.newCount || 0} NEW</span>
                        <button className="inline-action" onClick={() => refreshWatch([topic], `刷新 ${topic.query}`)} disabled={Boolean(refreshingWatch)}>{topicRefresh ? '刷新中' : '刷新'}</button>
                        <button className="inline-action secondary" onClick={() => useTopicForAnalysis(topic)} disabled={!topic.papers?.length}>用于分析</button>
                        <button className="inline-action secondary" onClick={() => clearSeenIds(topic)} title="只重置 NEW 演示状态，不删除论文。">重置 NEW 标记</button>
                        <button className="inline-action danger" onClick={() => removeWatchTopic(topic.id)}>移除</button>
                      </div>
                    </div>
                    <div className="meta-row">
                      <span>总计：{topic.totalCount || topic.papers?.length || 0} 篇</span>
                      <span>Semantic Scholar：{summary.semantic}</span>
                      <span>arXiv：{summary.arxiv}</span>
                      <span>最近来源：{latestRun ? sourceLabel(latestRun.sourceMode) : '尚未获取'}</span>
                      <span>时间范围：{latestRun ? timePresetLabel(latestRun.timePreset) : timePresetLabel(topic.timePreset)}</span>
                      <span>上次检查：{topic.checkedAt ? new Date(topic.checkedAt).toLocaleString() : '尚未刷新'}</span>
                      {topic.baselineCount ? <span>首次收录 {topic.baselineCount} 篇</span> : null}
                    </div>
                    {topicRefresh && <div className="job-status running"><strong>正在刷新</strong><span>{topicRefresh.label} · {topicElapsed}s</span></div>}
                    {!topicRefresh && topicStatus && <div className={`topic-status ${topic.error ? 'failed' : topic.sourceWarning && topic.papers?.length ? 'partial' : topic.totalCount ? 'completed' : 'empty-status'}`}>{topicStatus}</div>}
                    {topic.sourceWarning && <div className="warning compact">{topic.sourceWarning}</div>}
                    {topic.sortWarning && <div className="warning compact">{topic.sortWarning}</div>}
                    {topic.error && <div className="error compact">{topic.error}</div>}
                    {topic.runs?.length > 0 && (
                      <details className="run-history">
                        <summary>获取记录 · 当前显示 {visibleRuns.length} / 共 {topic.runs.length}</summary>
                        <div className="run-list">
                          {visibleRuns.map((run) => (
                            <div className={`run-item ${runStatusClass(run)}`} key={run.id}>
                              <strong>{runStatusLabel(run)}</strong>
                              <span>{sourceLabel(run.sourceMode)} · {run.sourceMode !== 'preprint' ? `${(run.venues || VENUE_GROUPS[run.venueGroup] || []).join(', ')} · ` : ''}{sortLabel(run.sortBy)} · {timePresetLabel(run.timePreset)} · 返回 {run.totalCount} 篇</span>
                              <small>{run.checkedAt ? new Date(run.checkedAt).toLocaleString() : '尚未刷新'}</small>
                            </div>
                          ))}
                        </div>
                        {topic.runs.length > 8 && (
                          <button className="inline-action secondary run-toggle" onClick={() => toggleRunsExpanded(topic.id)}>
                            {runsExpanded ? '收起记录' : `展开全部 ${topic.runs.length} 条记录`}
                          </button>
                        )}
                      </details>
                    )}
                    <div className="inbox-list">
                      {topicPapers.length > 0 && (
                        <div className="list-count">
                          当前显示 {visibleTopicPapers.length} / 总计 {topicPapers.length} 篇
                        </div>
                      )}
                      {visibleTopicPapers.map((paper) => (
                        <InboxPaperRow
                          key={`${topic.id}-${paper.id}`}
                          paper={paper}
                          onAdd={() => addToWorkbench(paper, topic)}
                          onIngest={() => handleIngestPdf(paper)}
                          onManualUpload={openManualUpload}
                          pdfJob={pdfJobs[paper.id]}
                          justAdded={Boolean(addedPaperIds[paper.id])}
                          isInWorkbench={workbenchIds.has(paperKey(paper))}
                        />
                      ))}
                      {topicPapers.length > 8 && (
                        <button className="inline-action secondary list-toggle" onClick={() => toggleTopicExpanded(topic.id)}>
                          {topicExpanded ? '收起论文' : `展开全部 ${topicPapers.length} 篇`}
                        </button>
                      )}
                      {!topic.papers?.length && !topic.error && <div className="empty">刷新后显示这个领域的论文；顶会模式若无结果会明确显示“无匹配论文”。</div>}
                    </div>
                  </article>
                  )
                })}
                {!watchTopics.length && (
                  <div className="empty-state">
                    <strong>先添加一个关注领域</strong>
                    <span>例如输入 `large language model agent` 或 `bert`，选择 `arXiv 预印本` 可以最快演示；选择 `顶会论文` 会使用 Semantic Scholar。</span>
                    <span>首次刷新作为 baseline 收录，不会把所有论文都标成 NEW；后续更新才突出新增论文。</span>
                  </div>
                )}
              </div>
            </section>

            <TracePanel trace={trace} visibleTrace={visibleTrace} traceExpanded={traceExpanded} setTraceExpanded={setTraceExpanded} />
          </>
        ) : (
          <section className="workbench-page">
            <div className="workbench-page-head">
              <div>
                <h3>工作台</h3>
                <span>{papers.length} 篇论文 · {selectedIds.length} 已选</span>
              </div>
            </div>
            <div className="workbench-tabs">
              <button className={drawerTab === 'list' ? 'active' : ''} onClick={() => setDrawerTab('list')}>论文列表</button>
              <button className={drawerTab === 'reading' ? 'active' : ''} onClick={() => setDrawerTab('reading')}>单篇精读</button>
              <button className={drawerTab === 'analysis' ? 'active' : ''} onClick={() => setDrawerTab('analysis')}>多论文分析</button>
            </div>

            {drawerTab === 'list' && (
              <div className="workbench-pane">
                <div className="section-title">
                  <div className="title-with-tip">
                    <h3>论文列表</h3>
                    <InfoTip text="从追踪收件箱加入的论文会进入这里，可勾选后用于排序、趋势、关系分析和文献综述。" />
                  </div>
                  <div className="button-row">
                    <SectionStatus job={sectionJobs.ranking} now={now} />
                    <button className="inline-action" onClick={handleRank} disabled={!papers.length || Boolean(loading)}>计算排序</button>
                    <button className="inline-action danger" onClick={clearWorkbench} disabled={!visiblePapers.length}>清空工作台</button>
                  </div>
                </div>
                <div className="paper-list">
                  {visiblePapers.map((paper) => (
                    <PaperCard
                      key={paper.id}
                      paper={paper}
                      selected={selectedIds.includes(paper.id)}
                      onToggle={() => togglePaper(paper.id)}
                      onIngest={() => handleIngestPdf(paper)}
                      onManualUpload={openManualUpload}
                      onRemove={() => removeFromWorkbench(paper.id)}
                      pdfJob={pdfJobs[paper.id]}
                    />
                  ))}
                  {!visiblePapers.length && <div className="empty-state"><strong>工作台还是空的</strong><span>从收件箱点击“加入工作台”后，论文会出现在这里。</span></div>}
                </div>
              </div>
            )}

            {drawerTab === 'reading' && (
              <div className="workbench-pane">
                <div className="section-title">
                  <div className="title-with-tip">
                    <h3>单篇论文精读</h3>
                    <InfoTip text="结构化总结、Reviewer 和问答都调用当前 .env 配置的 Kimi/OpenAI-compatible API；没有开放 PDF 时可手动上传 PDF。" />
                  </div>
                  <div className="button-row">
                    <button className="upload" onClick={() => uploadInputRef.current?.click()}>上传 PDF</button>
                  </div>
                </div>
                {uploaded ? (
                  <div className="upload-info">
                    <strong>{uploaded.filename}</strong>
                    <span>{uploaded.page_count} 页</span>
                    <span>{uploaded.chunk_count} chunks</span>
                    <span>{uploaded.char_count} 字符</span>
                    <PaperLinkBar paper={uploaded.sourcePaper} />
                  </div>
                ) : (
                  <div className="empty-state"><strong>还没有解析论文</strong><span>点击论文行里的“解析 PDF”，或上传本地 PDF 后继续总结、Reviewer 和问答。</span></div>
                )}
                <div className="button-row">
                  <button onClick={() => runLlmJob('summary', Boolean(summary))} disabled={!uploaded || summaryRunning}>{summary ? '重新生成总结' : '结构化总结'}{summaryRunning ? ` · ${llmJobs.summary?.elapsedSeconds || 0}s` : ''}</button>
                  <button onClick={() => runLlmJob('reviewer', Boolean(reviewer))} disabled={!uploaded || reviewerRunning}>{reviewer ? '重新生成 Reviewer' : 'Reviewer 视角'}{reviewerRunning ? ` · ${llmJobs.reviewer?.elapsedSeconds || 0}s` : ''}</button>
                </div>
                <ResultPanel title="结构化总结" content={summary} job={llmJobs.summary} mode="rendered" empty="解析 PDF 后生成结构化总结。">
                  {summary && (
                    <VisualAssetGallery
                      paperId={uploaded?.paper_id}
                      assets={visualAssets}
                    />
                  )}
                </ResultPanel>
                <ResultPanel title="Reviewer 分析" content={reviewer} job={llmJobs.reviewer} mode="rendered" empty="解析 PDF 后生成 Reviewer 分析。" />

                <div className="qa-panel">
                  <div className="section-title">
                    <div className="title-with-tip">
                      <h3>论文问答（已解析 PDF）</h3>
                      <InfoTip text="系统先从已解析 PDF 中检索依据片段，再调用 Kimi/OpenAI-compatible API 回答。" />
                    </div>
                    <SectionStatus job={sectionJobs.qa} now={now} />
                  </div>
                  <div className="qa-row">
                    <input value={question} onChange={(event) => setQuestion(event.target.value)} />
                    <button onClick={handleQa} disabled={!uploaded || Boolean(loading)}>提问</button>
                  </div>
                  {qaHistory.length > 0 && (
                    <div className="qa-history-panel">
                      <div className="section-title compact-title">
                        <strong>问答历史</strong>
                        <div className="button-row">
                          <span>{qaHistory.length} 条 · 已选 {selectedQaHistory.length}</span>
                          <button className="inline-action secondary" onClick={deleteSelectedQaHistory} disabled={!selectedQaHistory.length}>删除选中</button>
                          <button className="inline-action danger" onClick={clearCurrentQaHistory} disabled={!qaHistory.length}>清空问答</button>
                        </div>
                      </div>
                      <div className="qa-history-list">
                        {qaHistory.map((item) => (
                          <div className="qa-history-item" key={item.id}>
                            <input
                              type="checkbox"
                              checked={selectedQaHistory.includes(item.id)}
                              onChange={() => toggleQaHistory(item.id)}
                              aria-label={`选择问答 ${item.question}`}
                            />
                            <button
                              className="history-chip"
                              onClick={() => {
                                setQuestion(item.question)
                                setQa(item.payload)
                              }}
                            >
                              {item.question}
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {qa && (
                    <div className="qa-result">
                      <MarkdownBlock content={qa.answer} mode="rendered" />
                    </div>
                  )}
                </div>
                <div className="reading-history">
                  <div className="section-title">
                    <h3>精读记录</h3>
                    <div className="button-row">
                      <span>{readingRecords.length} 篇 · 已选 {selectedReadingRecords.length}</span>
                      <button className="inline-action secondary" onClick={deleteSelectedReadingRecords} disabled={!selectedReadingRecords.length}>删除选中</button>
                      <button className="inline-action danger" onClick={clearReadingRecords} disabled={!readingRecords.length}>清除全部</button>
                    </div>
                  </div>
                  <div className="reading-record-list">
                    {readingRecords.map((record) => (
                      <div
                        className={`reading-record ${uploaded?.paper_id === record.paper_id ? 'active' : ''}`}
                        key={record.paper_id}
                      >
                        <input
                          type="checkbox"
                          checked={selectedReadingRecords.includes(record.paper_id)}
                          onChange={() => toggleReadingRecord(record.paper_id)}
                          aria-label={`选择精读记录 ${record.filename}`}
                        />
                        <button className="reading-record-open" onClick={() => openReadingRecord(record)}>
                          <strong>{record.filename}</strong>
                          <span>{record.page_count || 0} 页 · {record.chunk_count || 0} chunks · {record.updatedAt ? new Date(record.updatedAt).toLocaleString() : ''}</span>
                          <small>
                            {record.summary ? '总结' : '未总结'} · {record.reviewer ? 'Reviewer' : '未生成 Reviewer'} · {record.qaCount || record.qaHistory?.length || 0} 条问答
                          </small>
                        </button>
                      </div>
                    ))}
                    {!readingRecords.length && <div className="empty">解析 PDF 后，这里会保存精读记录；下次可以直接点开以前的总结、Reviewer 和问答。</div>}
                  </div>
                </div>
              </div>
            )}

            {drawerTab === 'analysis' && (
              <div className="workbench-pane">
                <div className="analysis-scope">
                  <div>
                    <strong>{analysisFields.length > 1 ? '跨领域分析' : '分析范围'}</strong>
                    <span>
                      当前使用 {analysisPapers.length} 篇论文
                      {analysisFields.length > 1 ? `，来自 ${analysisFields.join('、')}。综述会按领域分组比较。` : (analysisFields[0] ? `，领域：${analysisFields[0]}。` : '。')}
                    </span>
                  </div>
                  {workbenchFields.length > 1 && (
                    <label>
                      <span>领域筛选</span>
                      <select value={analysisFieldFilter} onChange={(event) => setAnalysisFieldFilter(event.target.value)}>
                        <option value="ALL">全部领域</option>
                        {workbenchFields.map((field) => <option key={field} value={field}>{field}</option>)}
                      </select>
                    </label>
                  )}
                </div>
                <div className="two-column">
                  <section className="analysis-block">
                    <div className="section-title">
                      <h3>研究趋势</h3>
                      <div className="button-row">
                        <SectionStatus job={sectionJobs.trends} now={now} />
                        <button className="inline-action" onClick={handleTrends} disabled={!analysisPapers.length || Boolean(loading)}>趋势分析</button>
                        <span>{trends?.topics?.length || 0} topics</span>
                      </div>
                    </div>
                    {trends ? (
                      <>
                        <div className="topic-grid">
                          {trends.topics.map((topic) => (
                            <div className="topic" key={topic.name}>
                              <strong>{topic.name}</strong>
                              <span>{topic.paper_count} 篇</span>
                            </div>
                          ))}
                        </div>
                        <MarkdownBlock content={trends.llm_analysis} mode={markdownMode} />
                      </>
                    ) : (
                      <div className="empty">将论文加入工作台后生成趋势分析。</div>
                    )}
                  </section>

                  <section className="analysis-block">
                    <div className="section-title">
                      <div className="title-with-tip">
                        <h3>论文关系分析</h3>
                        <InfoTip text="基于标题和摘要 embedding，发现主题相近、方法路线接近的论文；这不是引用网络。" />
                      </div>
                      <div className="button-row">
                        <SectionStatus job={sectionJobs.graph} now={now} />
                        <button className="inline-action" onClick={handleGraph} disabled={!analysisPapers.length || Boolean(loading)}>关系分析</button>
                        <span>{graph.edges.length} edges</span>
                      </div>
                    </div>
                    <GraphView graph={graph} />
                  </section>
                </div>

                <section className="analysis-block">
                  <div className="section-title">
                    <h3>自动文献综述</h3>
                    <div className="button-row">
                      <SectionStatus job={sectionJobs.literature} now={now} />
                      <button className="inline-action" onClick={handleLiteratureReview} disabled={!analysisPapers.length || Boolean(loading)}>生成综述</button>
                      <span>{selectedPapers.length} selected</span>
                    </div>
                  </div>
                  {literatureReview ? <MarkdownBlock content={literatureReview} mode={markdownMode} /> : <div className="empty">勾选工作台论文后生成综述。</div>}
                </section>
              </div>
            )}
          </section>
        )}
      </main>
      <button
        className={`workbench-handle ${viewMode === 'workbench' ? 'return-mode' : ''} ${drawerPulse ? 'pulse' : ''}`}
        onClick={() => viewMode === 'workbench' ? returnToInbox() : openWorkbench('list')}
        aria-label={viewMode === 'workbench' ? '返回收件箱' : '打开工作台'}
      >
        <span>{viewMode === 'workbench' ? '返回' : '工作台'}</span>
        <strong>{viewMode === 'workbench' ? '←' : papers.length}</strong>
      </button>
      <input ref={uploadInputRef} className="visually-hidden-file" type="file" accept="application/pdf" onChange={handleUpload} />
    </div>
  )
}

function InfoTip({ text }) {
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const ref = useRef(null)
  const show = () => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    const width = Math.min(340, window.innerWidth - 24)
    const left = Math.min(Math.max(rect.left + rect.width / 2 - width / 2, 12), window.innerWidth - width - 12)
    const top = rect.top > 110 ? rect.top - 12 : rect.bottom + 12
    setPosition({ top, left, width, placement: rect.top > 110 ? 'top' : 'bottom' })
    setOpen(true)
  }
  const hide = () => setOpen(false)
  return (
    <>
    <button
      ref={ref}
      type="button"
      className="info-tip"
      aria-label={text}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      onClick={() => (open ? hide() : show())}
    >
      ?
    </button>
    {open && createPortal(
      <div
        className={`tooltip-layer ${position.placement === 'bottom' ? 'below' : ''}`}
        role="tooltip"
        style={{ top: position.top, left: position.left, width: position.width }}
      >
        {text}
      </div>,
      document.body,
    )}
    </>
  )
}

function TracePanel({ trace, visibleTrace, traceExpanded, setTraceExpanded }) {
  return (
    <section className="section trace-panel">
      <div className="section-title">
        <div className="title-with-tip">
          <h3>Agent 执行轨迹</h3>
          <InfoTip text="默认显示最新几条检索、排序、PDF 解析和 LLM 调用步骤；展开后可检查完整运行过程。" />
        </div>
        <div className="trace-actions">
          <span>{trace.length} steps</span>
          {trace.length > 8 && (
            <button className="inline-action secondary" onClick={() => setTraceExpanded((value) => !value)}>
              {traceExpanded ? '收起' : `展开全部 ${trace.length}`}
            </button>
          )}
        </div>
      </div>
      <div className="trace-list">
        {visibleTrace.map((step, index) => (
          <div className={`trace ${step.status}`} key={`${step.group}-${step.name}-${index}`}>
            <strong>{step.group} · {step.name}</strong>
            <span>{step.status} · {step.elapsed_ms} ms · {step.stamp}</span>
            <p>{step.detail}</p>
          </div>
        ))}
        {!trace.length && <div className="empty-state"><strong>暂无执行轨迹</strong><span>检索论文、计算排序、解析 PDF 或生成内容后，系统会在这里留下可检查的过程记录。</span></div>}
      </div>
    </section>
  )
}

function sourceLabel(sourceMode) {
  if (sourceMode === 'conference') return '顶会论文'
  if (sourceMode === 'hybrid') return '顶会 + arXiv'
  return 'arXiv 预印本'
}

function paperExternalLabel(paper) {
  const source = String(paper.source || '')
  if (source.includes('arXiv') || paper.arxiv_url?.includes('arxiv.org')) return 'arXiv'
  if (source.includes('Semantic Scholar') || paper.paper_url?.includes('semanticscholar.org')) return 'Semantic Scholar'
  return '论文页面'
}

function paperSourceLinks(paper) {
  const links = []
  if (paper.paper_url?.includes('semanticscholar.org')) {
    links.push({ label: 'Semantic Scholar', url: paper.paper_url })
  }
  if (paper.arxiv_url?.includes('arxiv.org')) {
    links.push({ label: 'arXiv', url: paper.arxiv_url })
  }
  if (!links.length) {
    const fallback = paper.paper_url || paper.arxiv_url
    if (fallback) links.push({ label: paperExternalLabel(paper), url: fallback })
  }
  return links.filter((item, index, array) => array.findIndex((other) => other.url === item.url) === index)
}

function scoreBreakdownItems(paper) {
  return [
    ['语义相关性', paper.relevance_score, '使用 sentence-transformers 计算论文标题/摘要与当前领域关键词的语义相似度。'],
    ['时间新近性', paper.recency_score, '根据发布日期换算，越新的论文分数越高。'],
    ['关键词匹配', paper.keyword_score, '根据标题和摘要中是否直接命中领域关键词计算。'],
    ['代码线索', paper.code_score, '根据摘要中是否出现 code、github、repository 等开放实现线索计算。'],
    ['引用量', paper.citation_score, '来自 Semantic Scholar 的 citationCount 归一化；arXiv 论文通常没有该项。'],
  ].filter(([, value]) => value !== undefined && value !== null)
}

function sortLabel(sortBy) {
  if (sortBy === 'latest') return '最新优先'
  if (sortBy === 'citations') return '引用量优先'
  return '综合推荐'
}

function runStatusLabel(run) {
  if (run.error) return '刷新失败'
  if (run.sourceWarning) return '部分补充完成'
  return run.statusMessage || '刷新完成'
}

function runStatusClass(run) {
  if (run.error) return 'failed'
  if (run.sourceWarning) return 'partial'
  if (run.totalCount) return 'completed'
  return 'empty-status'
}

function InboxPaperRow({ paper, onAdd, onIngest, onManualUpload, pdfJob, justAdded, isInWorkbench }) {
  const isRunning = pdfJob && !['completed', 'failed'].includes(pdfJob.status)
  const sourceLinks = paperSourceLinks(paper)
  return (
    <div className={`inbox-paper ${paper.is_new ? 'new-paper' : ''} ${justAdded ? 'just-added' : ''}`}>
      <div>
        <div className="title-row">
          <strong>{paper.title}</strong>
          {paper.is_new && <span className="new-pill">NEW</span>}
        </div>
        <div className="meta-row">
          <span>{paper.venue || paper.source}</span>
          <span>{paper.year || paper.published?.slice(0, 10)}</span>
          <span>{paper.source}</span>
          {paper.citation_count ? <span>引用 {paper.citation_count}</span> : null}
          {paper.score !== undefined && <span>score {paper.score.toFixed(3)}</span>}
        </div>
      </div>
      <div className="watch-actions">
        {sourceLinks.map((link) => <a key={link.url} href={link.url} target="_blank" rel="noreferrer">{link.label}</a>)}
        {paper.pdf_url && <a href={paper.pdf_url} target="_blank" rel="noreferrer">PDF</a>}
        <button className={`inline-action secondary ${isInWorkbench ? 'added' : ''}`} onClick={onAdd} disabled={isInWorkbench}>
          {isInWorkbench ? '已加入' : '加入工作台'}
        </button>
        {paper.pdf_url ? (
          <button className="inline-action" onClick={onIngest} disabled={isRunning}>{isRunning ? '解析中' : '解析 PDF'}</button>
        ) : (
          <>
            <span className="pdf-unavailable">无开放 PDF</span>
            <button className="inline-action" onClick={onManualUpload}>上传 PDF 精读</button>
          </>
        )}
      </div>
      {pdfJob && <JobStatus job={pdfJob} />}
    </div>
  )
}

function PaperCard({ paper, selected, onToggle, onIngest, onManualUpload, onRemove, pdfJob }) {
  const isRunning = pdfJob && !['completed', 'failed'].includes(pdfJob.status)
  const sourceLinks = paperSourceLinks(paper)
  const scoreItems = scoreBreakdownItems(paper)
  return (
    <article className="paper-card">
      <div className="paper-card-head">
        <div className="select-cell">
          <input type="checkbox" checked={selected} onChange={onToggle} aria-label={`选择 ${paper.title}`} />
        </div>
        <div className="paper-card-main">
          <h4>{paper.title}</h4>
          <p>{paper.authors?.slice(0, 5).join(', ')}</p>
        </div>
        <div className="paper-card-side">
          {paper.score !== undefined && <span className="score">{paper.score.toFixed(3)}</span>}
          <button className="inline-action danger" onClick={onRemove}>移除</button>
        </div>
      </div>
      <p className="abstract">{paper.summary}</p>
      <div className="meta-row">
        <span>{paper.published?.slice(0, 10) || paper.year}</span>
        <span>{paper.venue || paper.categories?.join(', ')}</span>
        <span>{paper.source}</span>
        {(paper.sourceField || paper.source_field) && <span>来源领域：{paper.sourceField || paper.source_field}</span>}
        {paper.citation_count ? <span>引用 {paper.citation_count}</span> : null}
        {sourceLinks.map((link) => <a key={link.url} href={link.url} target="_blank" rel="noreferrer">{link.label}</a>)}
        {paper.pdf_url && <a href={paper.pdf_url} target="_blank" rel="noreferrer">PDF</a>}
        {paper.pdf_url ? (
          <button className="inline-action" onClick={onIngest} disabled={isRunning}>
            {isRunning ? '解析中' : '解析 PDF'}
          </button>
        ) : (
          <>
            <span className="pdf-unavailable">无开放 PDF</span>
            <button className="inline-action" onClick={onManualUpload}>上传 PDF 精读</button>
          </>
        )}
      </div>
      {pdfJob && <JobStatus job={pdfJob} />}
      {paper.explanation && (
        <div className="score-explain">
          <div className="score-explain-head">
            <strong>算法排序评分</strong>
            <InfoTip text="这个分数不是 AI 主观评价，而是由语义相关性、发布时间、关键词匹配、代码线索和引用量等规则综合计算。" />
          </div>
          <div className="score-grid">
            {scoreItems.map(([label, value, tip]) => (
              <span key={label} title={tip}>{label} {Number(value).toFixed(2)}</span>
            ))}
          </div>
          <small>{paper.explanation}</small>
        </div>
      )}
    </article>
  )
}

function JobStatus({ job }) {
  return (
    <div className={`job-status ${job.status}`}>
      <strong>{job.stage}</strong>
      <span>{job.status}{job.elapsedSeconds ? ` · ${job.elapsedSeconds}s` : ''}</span>
      {job.error && <p>{job.error}</p>}
      {job.status === 'completed' && job.result?.page_count && (
        <p>已解析：{job.result.page_count} 页，{job.result.chunk_count} chunks，可在下方精读区继续分析。</p>
      )}
      {job.trace?.length > 0 && (
        <div className="job-steps">
          {job.trace.map((step, index) => (
            <span key={`${step.name}-${index}`} className={step.status}>{step.name}</span>
          ))}
        </div>
      )}
    </div>
  )
}

function SectionStatus({ job, now }) {
  if (!job || job.status === 'completed') return null
  const elapsed = job.startedAt ? Math.round(((job.completedAt || now) - job.startedAt) / 1000) : 0
  return (
    <span className={`section-status ${job.status}`}>
      {job.status === 'running' ? `${job.label}中 · ${elapsed}s` : `${job.label}失败`}
      {job.error ? `：${job.error}` : ''}
    </span>
  )
}

function ResultPanel({ title, content, job, mode, empty, children }) {
  return (
    <div className="result-panel">
      <div className="result-head">
        <strong>{title}</strong>
        {job && <span>{job.status} · {job.stage}{job.elapsedSeconds ? ` · ${job.elapsedSeconds}s` : ''}</span>}
      </div>
      {job?.error && <div className="error compact">{job.error}</div>}
      {content ? <MarkdownBlock content={content} mode={mode} /> : <div className="empty">{empty}</div>}
      {children}
    </div>
  )
}

function MarkdownBlock({ content, mode }) {
  const normalized = normalizeMarkdownContent(content)
  if (mode === 'source') {
    return <pre className="markdown-block">{normalized}</pre>
  }
  return (
    <div className="markdown-rendered">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalized}</ReactMarkdown>
    </div>
  )
}

function normalizeMarkdownContent(content) {
  let text = String(content || '').trim()
  const fenced = text.match(/^```(?:markdown|md|text)?\s*([\s\S]*?)\s*```$/i)
  if (fenced) text = fenced[1].trim()
  const lines = text.split('\n')
  const nonEmpty = lines.filter((line) => line.trim())
  if (nonEmpty.length && nonEmpty.every((line) => /^ {2,}/.test(line))) {
    const minIndent = Math.min(...nonEmpty.map((line) => line.match(/^ */)[0].length))
    text = lines.map((line) => line.slice(Math.min(minIndent, line.match(/^ */)[0].length))).join('\n')
  }
  return text.replace(/\n{3,}/g, '\n\n')
}

function evidenceSummary(text) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim()
  if (normalized.length <= 170) return normalized
  const sentences = normalized.split(/(?<=[.!?。！？])\s+/).filter(Boolean)
  const summary = sentences.slice(0, 2).join(' ')
  return (summary || normalized).slice(0, 210)
}

function highlightEvidence(text, question) {
  const words = Array.from(new Set(String(question || '')
    .toLowerCase()
    .split(/[^a-zA-Z0-9\u4e00-\u9fa5]+/)
    .filter((word) => word.length >= 3)
    .slice(0, 8)))
  if (!words.length) return text
  const escaped = words.map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const pattern = new RegExp(`(${escaped.join('|')})`, 'gi')
  return text.split(pattern).map((part, idx) => {
    const matched = words.includes(part.toLowerCase())
    return matched ? <mark key={`${part}-${idx}`}>{part}</mark> : part
  })
}

function EvidenceCard({ item, index, question }) {
  const [expanded, setExpanded] = useState(false)
  const text = item.text || ''
  const summary = evidenceSummary(text)
  const showToggle = text.length > summary.length + 20
  return (
    <div className={`evidence ${expanded ? 'expanded' : ''}`}>
      <div className="evidence-title">
        <strong>依据 {index + 1} · 第 {item.page} 页</strong>
        <span>相关度 {item.score.toFixed(3)}</span>
      </div>
      <p className="evidence-summary">{summary}</p>
      {expanded && <p className="evidence-full">{highlightEvidence(text, question)}</p>}
      {showToggle && (
        <button className="inline-action secondary" onClick={() => setExpanded((value) => !value)}>
          {expanded ? '收起原文' : '展开原文'}
        </button>
      )}
    </div>
  )
}

function PaperLinkBar({ paper }) {
  const links = paper ? paperSourceLinks(paper) : []
  if (!links.length) {
    return <span className="source-links local-pdf">本地上传 PDF</span>
  }
  return (
    <span className="source-links">
      {links.map((link) => (
        <a href={link.url} target="_blank" rel="noreferrer" key={link.label}>{link.label}</a>
      ))}
    </span>
  )
}

function VisualAssetGallery({ paperId, assets }) {
  const visibleAssets = (assets || []).filter((asset) => asset?.asset_id)
  if (!paperId) return null
  const sourceText = (asset) => {
    if (asset.source === 'kimi-selected') return 'Kimi k2.6 判别'
    return 'fallback'
  }
  return (
    <div className="visual-assets">
      <div className="visual-assets-head">
        <strong>图表辅助解读</strong>
        <span>系统先生成候选裁剪图，再复用 Kimi k2.6 判断技术架构图和主结果表；未选中时会显示低置信 caption 兜底候选。</span>
      </div>
      {!visibleAssets.length ? (
        <div className="empty compact">
          未识别到明确的技术架构图或主结果表；系统不会编造图表内容。
        </div>
      ) : (
        <div className="visual-asset-grid">
          {visibleAssets.map((asset) => (
            <figure className="visual-asset" key={asset.asset_id}>
              <a
                href={`${API_BASE}/api/papers/${paperId}/visual-assets/${asset.asset_id}/image`}
                target="_blank"
                rel="noreferrer"
                title="打开裁剪图"
              >
                <img
                  src={`${API_BASE}/api/papers/${paperId}/visual-assets/${asset.asset_id}/image`}
                  alt={asset.label || '论文图表裁剪图'}
                  loading="lazy"
                />
              </a>
              <figcaption>
                <strong>{asset.label || (asset.asset_type === 'architecture' ? '技术架构图' : '主结果表')}</strong>
                <span>
                  第 {asset.page} 页 · {asset.asset_type === 'architecture' ? '技术架构图' : '主结果表'}
                  {' · '}{sourceText(asset)}
                  {asset.confidence ? ` · ${asset.confidence}` : ''}
                  {asset.candidate_count ? ` · 候选 ${asset.candidate_count} 个` : ''}
                </span>
                {(asset.vision_reason || asset.reason) && <p>{asset.vision_reason || asset.reason}</p>}
                {asset.caption && <p>{asset.caption}</p>}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  )
}

function VisualPagePreview({ paperId, pages }) {
  const validPages = (pages || [])
    .filter((item) => item?.page && Number(item.page) > 0)
    .slice(0, 3)
  if (!paperId || !validPages.length) return null
  return (
    <div className="visual-pages">
      <div className="visual-pages-head">
        <strong>原论文图表/页面截图</strong>
        <span>以下页面由已解析 PDF 直接渲染，可用来核对答案中提到的图、表或方法框架。</span>
      </div>
      <div className="visual-page-grid">
        {validPages.map((item) => (
          <figure className="visual-page" key={`${item.page}-${item.label || ''}`}>
            <a
              href={`${API_BASE}/api/papers/${paperId}/pages/${item.page}/image?zoom=2`}
              target="_blank"
              rel="noreferrer"
              title="打开大图"
            >
              <img
                src={`${API_BASE}/api/papers/${paperId}/pages/${item.page}/image?zoom=1.15`}
                alt={`原论文第 ${item.page} 页截图`}
                loading="lazy"
              />
            </a>
            <figcaption>
              <strong>{item.label || `第 ${item.page} 页`}</strong>
              <span>第 {item.page} 页</span>
              {item.reason && <p>{item.reason}</p>}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  )
}

function GraphView({ graph }) {
  if (!graph.nodes.length) return <div className="empty">先把多篇论文加入工作台，再生成关系分析；系统会展示论文之间的主题和方法相似关系。</div>
  const nodeTitles = Object.fromEntries(graph.nodes.map((node, index) => [node.id, `${index + 1}. ${node.title}`]))
  const width = 520
  const height = 340
  const cx = width / 2
  const cy = height / 2
  const radius = 130
  const positions = Object.fromEntries(graph.nodes.map((node, index) => {
    const angle = (Math.PI * 2 * index) / graph.nodes.length
    return [node.id, { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius }]
  }))

  return (
    <div className="graph-wrap">
      <div>
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="论文关系分析图">
          {graph.edges.map((edge) => {
            const source = positions[edge.source]
            const target = positions[edge.target]
            if (!source || !target) return null
            return <line key={`${edge.source}-${edge.target}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} strokeWidth={1 + edge.weight * 4} />
          })}
          {graph.nodes.map((node, index) => {
            const pos = positions[node.id]
            return (
              <g key={node.id}>
                <circle cx={pos.x} cy={pos.y} r="13" />
                <text x={pos.x + 16} y={pos.y + 4}>{index + 1}</text>
                <title>{node.title}</title>
              </g>
            )
          })}
        </svg>
        <div className="node-list">
          <strong>编号对照 · 共 {graph.nodes.length} 篇</strong>
          <ol>
            {graph.nodes.map((node, index) => <li key={node.id} value={index + 1}>{node.title}</li>)}
          </ol>
        </div>
      </div>
      <div className="edge-list">
        <strong>Top 相似关系</strong>
        {graph.edges.slice(0, 8).map((edge, index) => (
          <div className="edge-card" key={`${edge.source}-${edge.target}`}>
            <span>关系 {index + 1} · 相似度 {edge.weight.toFixed(3)}</span>
            <p>{nodeTitles[edge.source]} ↔ {nodeTitles[edge.target]}</p>
            <small>{edge.reason}。可用于识别同一主题簇，并辅助趋势分析和文献综述。</small>
          </div>
        ))}
        {!graph.edges.length && <div className="empty">当前论文之间没有超过阈值的强相似关系，可降低阈值或加入更多论文。</div>}
      </div>
    </div>
  )
}

export default App
