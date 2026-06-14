from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Optional
from uuid import uuid4

from .schemas import TraceStep


@dataclass
class PdfJob:
    job_id: str
    status: str = "queued"
    stage: str = "等待中"
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    content: str = ""
    trace: list[TraceStep] = field(default_factory=list)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, PdfJob] = {}
        self._lock = threading.Lock()

    def create(self) -> PdfJob:
        job = PdfJob(job_id=str(uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> PdfJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(f"Job not found: {job_id}")
            return PdfJob(
                job_id=job.job_id,
                status=job.status,
                stage=job.stage,
                error=job.error,
                result=job.result,
                content=job.content,
                trace=list(job.trace),
            )

    def start_step(self, job_id: str, name: str, detail: str = "") -> float:
        started = time.perf_counter()
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.stage = name
            job.error = None
            job.trace.append(TraceStep(name=name, status="running", detail=detail))
        return started

    def finish_step(self, job_id: str, started: float) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.trace:
                current = job.trace[-1]
                current.status = "completed"
                current.elapsed_ms = int((time.perf_counter() - started) * 1000)

    def fail_step(self, job_id: str, started: float, error: Exception) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.stage = "失败"
            job.error = str(error)
            if job.trace:
                current = job.trace[-1]
                current.status = "failed"
                current.detail = str(error)
                current.elapsed_ms = int((time.perf_counter() - started) * 1000)

    def append_content(self, job_id: str, content: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.content += content

    def complete(self, job_id: str, result: dict[str, Any] | None = None, content: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.stage = "完成"
            job.result = result
            if content is not None:
                job.content = content
            job.error = None


job_store = JobStore()
