"""
Latency measurement helpers.
Each stage timer is a lightweight async context manager that records
start/end and writes a structured log entry.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyReport:
    session_id: str
    request_id: str
    stages: Dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, elapsed_ms: float) -> None:
        self.stages[stage] = round(elapsed_ms, 2)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 2)

    def log(self) -> None:
        logger.info(
            "Pipeline latency report",
            extra={
                "session_id": self.session_id,
                "request_id": self.request_id,
                **{f"latency_{k}_ms": v for k, v in self.stages.items()},
                "latency_total_ms": self.total_ms,
            },
        )


@asynccontextmanager
async def measure(
    report: LatencyReport, stage: str
) -> AsyncIterator[None]:
    """Async context manager that records elapsed time for a named stage."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        report.record(stage, elapsed)
        logger.debug(
            f"Stage '{stage}' completed",
            extra={
                "session_id": report.session_id,
                "request_id": report.request_id,
                "stage": stage,
                "elapsed_ms": round(elapsed, 2),
            },
        )
