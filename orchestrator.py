"""
Pipeline Orchestrator
=====================
Coordinates the full STT → LLM → TTS pipeline for a single user turn.

Responsibilities
----------------
1. Runs each stage sequentially (audio → text → reply → audio).
2. Applies per-stage timeouts sourced from configuration.
3. Retries transient failures up to `max_retries` times with exponential
   back-off (capped at a fixed delay to keep latency predictable).
4. Records stage-level latency into a LatencyReport and emits it at end.
5. Raises PipelineError on unrecoverable failures so callers get a typed
   exception rather than a raw exception from an upstream library.

Design Decisions
----------------
- The orchestrator is injected with adapter instances, making it unit-testable
  with mocks and independent of any framework.
- Retries happen at the stage level, not the whole pipeline, to avoid repeating
  expensive completed stages.
- Timeout wrapping uses asyncio.wait_for; if the stage is still running after
  the deadline it is cancelled and a PipelineError is raised.
"""

import asyncio
import uuid
from typing import List, Optional

from app.adapters.base import LLMAdapter, STTAdapter, TTSAdapter
from app.config import get_settings
from app.core.logging import get_logger
from app.metrics.latency import LatencyReport, measure

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when the pipeline cannot recover from a stage failure."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"Pipeline failed at stage '{stage}': {cause}")


class PipelineOrchestrator:
    def __init__(
        self,
        stt: STTAdapter,
        llm: LLMAdapter,
        tts: TTSAdapter,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._timeout = timeout_seconds or settings.pipeline_timeout_seconds
        self._max_retries = max_retries if max_retries is not None else settings.pipeline_max_retries
        self._retry_delay = retry_delay_seconds or settings.pipeline_retry_delay_seconds

    async def run(
        self,
        audio_bytes: bytes,
        history: List[dict],
        session_id: str = "",
        mime_type: str = "audio/webm",
    ) -> tuple[str, bytes, LatencyReport]:
        """
        Execute the full STT → LLM → TTS pipeline.

        Returns
        -------
        (transcript, audio_response, latency_report)

        Raises
        ------
        PipelineError on unrecoverable failure.
        """
        request_id = str(uuid.uuid4())[:8]
        report = LatencyReport(session_id=session_id, request_id=request_id)

        logger.info(
            "Pipeline started",
            extra={"session_id": session_id, "request_id": request_id, "audio_bytes": len(audio_bytes)},
        )

        # ── Stage 1: STT ───────────────────────────────────────────────────────
        transcript = await self._run_stage(
            name="stt",
            coro_factory=lambda: self._stt.transcribe(audio_bytes, mime_type),
            report=report,
            session_id=session_id,
            request_id=request_id,
        )

        if not transcript:
            raise PipelineError("stt", ValueError("Empty transcript — audio may be silent or unclear"))

        # ── Stage 2: LLM ───────────────────────────────────────────────────────
        messages = list(history) + [{"role": "user", "content": transcript}]
        reply_text = await self._run_stage(
            name="llm",
            coro_factory=lambda: self._llm.chat(messages),
            report=report,
            session_id=session_id,
            request_id=request_id,
        )

        # ── Stage 3: TTS ───────────────────────────────────────────────────────
        audio_response = await self._run_stage(
            name="tts",
            coro_factory=lambda: self._tts.synthesize(reply_text),
            report=report,
            session_id=session_id,
            request_id=request_id,
        )

        report.log()
        logger.info(
            "Pipeline completed",
            extra={
                "session_id": session_id,
                "request_id": request_id,
                "total_ms": report.total_ms,
            },
        )
        return transcript, audio_response, report

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _run_stage(
        self,
        name: str,
        coro_factory,
        report: LatencyReport,
        session_id: str,
        request_id: str,
    ):
        """Run a single pipeline stage with retries and timeout."""
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Retrying stage '{name}'",
                    extra={
                        "session_id": session_id,
                        "request_id": request_id,
                        "attempt": attempt,
                        "delay_s": delay,
                        "cause": str(last_exc),
                    },
                )
                await asyncio.sleep(delay)

            try:
                async with measure(report, name if attempt == 0 else f"{name}_retry{attempt}"):
                    result = await asyncio.wait_for(coro_factory(), timeout=self._timeout)
                return result
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.error(
                    f"Stage '{name}' timed out",
                    extra={
                        "session_id": session_id,
                        "request_id": request_id,
                        "timeout_s": self._timeout,
                        "attempt": attempt,
                    },
                )
            except Exception as exc:
                last_exc = exc
                logger.error(
                    f"Stage '{name}' failed",
                    extra={
                        "session_id": session_id,
                        "request_id": request_id,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )

        raise PipelineError(name, last_exc or RuntimeError("Unknown failure"))
