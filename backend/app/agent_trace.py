import time
from contextlib import contextmanager
from typing import Iterator

from .schemas import TraceStep


class AgentTrace:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    @contextmanager
    def step(self, name: str, detail: str = "") -> Iterator[None]:
        started = time.perf_counter()
        current = TraceStep(name=name, status="running", detail=detail)
        self.steps.append(current)
        try:
            yield
        except Exception as exc:
            current.status = "failed"
            current.detail = str(exc)
            current.elapsed_ms = int((time.perf_counter() - started) * 1000)
            raise
        else:
            current.status = "completed"
            current.elapsed_ms = int((time.perf_counter() - started) * 1000)

    def add(self, name: str, detail: str, status: str = "completed", elapsed_ms: int = 0) -> None:
        self.steps.append(TraceStep(name=name, status=status, detail=detail, elapsed_ms=elapsed_ms))
