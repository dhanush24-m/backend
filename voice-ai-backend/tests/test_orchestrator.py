"""
Unit tests for PipelineOrchestrator.

All external adapters are mocked; no real API calls are made.
Tests assert behaviour, not just that code ran without errors.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline.orchestrator import PipelineError, PipelineOrchestrator


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_orchestrator(stt_mock, llm_mock, tts_mock, **kwargs):
    return PipelineOrchestrator(
        stt=stt_mock,
        llm=llm_mock,
        tts=tts_mock,
        timeout_seconds=kwargs.get("timeout_seconds", 5.0),
        max_retries=kwargs.get("max_retries", 1),
        retry_delay_seconds=kwargs.get("retry_delay_seconds", 0.01),
    )


def make_adapters(transcript="Hello world", reply="Hi there!", audio=b"\xFF\xD8\xFF"):
    stt = AsyncMock()
    stt.transcribe.return_value = transcript
    llm = AsyncMock()
    llm.chat.return_value = reply
    tts = AsyncMock()
    tts.synthesize.return_value = audio
    return stt, llm, tts


# ── Happy path ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_pipeline_returns_transcript_and_audio():
    stt, llm, tts = make_adapters(transcript="test input", reply="test reply", audio=b"audio_data")
    orch = make_orchestrator(stt, llm, tts)

    transcript, audio, report = await orch.run(
        audio_bytes=b"raw_audio",
        history=[],
        session_id="test-session",
    )

    assert transcript == "test input"
    assert audio == b"audio_data"


@pytest.mark.asyncio
async def test_pipeline_passes_history_to_llm():
    stt, llm, tts = make_adapters(transcript="follow-up question")
    existing_history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    orch = make_orchestrator(stt, llm, tts)

    await orch.run(audio_bytes=b"audio", history=existing_history, session_id="sess")

    called_messages = llm.chat.call_args[0][0]
    # History messages should appear before the new user message
    assert called_messages[-3]["content"] == "first question"
    assert called_messages[-2]["content"] == "first answer"
    assert called_messages[-1]["role"] == "user"
    assert called_messages[-1]["content"] == "follow-up question"


@pytest.mark.asyncio
async def test_latency_report_contains_all_stages():
    stt, llm, tts = make_adapters()
    orch = make_orchestrator(stt, llm, tts)

    _, _, report = await orch.run(audio_bytes=b"audio", history=[], session_id="s")

    assert "stt" in report.stages
    assert "llm" in report.stages
    assert "tts" in report.stages
    assert report.total_ms > 0


# ── Empty transcript ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_transcript_raises_pipeline_error():
    stt, llm, tts = make_adapters(transcript="")  # Silence / no speech detected
    orch = make_orchestrator(stt, llm, tts, max_retries=0)

    with pytest.raises(PipelineError) as exc_info:
        await orch.run(audio_bytes=b"silent_audio", history=[], session_id="s")

    assert exc_info.value.stage == "stt"


# ── Retries ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stt_retried_on_failure_then_succeeds():
    stt = AsyncMock()
    stt.transcribe.side_effect = [RuntimeError("network blip"), "retry succeeded"]
    llm = AsyncMock()
    llm.chat.return_value = "ok"
    tts = AsyncMock()
    tts.synthesize.return_value = b"audio"

    orch = make_orchestrator(stt, llm, tts, max_retries=1, retry_delay_seconds=0.01)

    transcript, _, _ = await orch.run(audio_bytes=b"audio", history=[], session_id="s")

    assert transcript == "retry succeeded"
    assert stt.transcribe.call_count == 2


@pytest.mark.asyncio
async def test_llm_fails_after_all_retries_raises_pipeline_error():
    stt = AsyncMock()
    stt.transcribe.return_value = "hello"
    llm = AsyncMock()
    llm.chat.side_effect = RuntimeError("LLM down")
    tts = AsyncMock()
    tts.synthesize.return_value = b"audio"

    orch = make_orchestrator(stt, llm, tts, max_retries=2, retry_delay_seconds=0.01)

    with pytest.raises(PipelineError) as exc_info:
        await orch.run(audio_bytes=b"audio", history=[], session_id="s")

    assert exc_info.value.stage == "llm"
    assert llm.chat.call_count == 3  # initial + 2 retries


# ── Timeout ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tts_timeout_raises_pipeline_error():
    stt = AsyncMock()
    stt.transcribe.return_value = "hello"
    llm = AsyncMock()
    llm.chat.return_value = "hi"
    tts = AsyncMock()

    async def slow_tts(*_):
        await asyncio.sleep(10)  # Will be cancelled by timeout

    tts.synthesize.side_effect = slow_tts

    orch = make_orchestrator(stt, llm, tts, timeout_seconds=0.05, max_retries=0)

    with pytest.raises(PipelineError) as exc_info:
        await orch.run(audio_bytes=b"audio", history=[], session_id="s")

    assert exc_info.value.stage == "tts"


# ── Stage isolation ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tts_not_called_when_llm_fails():
    stt = AsyncMock()
    stt.transcribe.return_value = "hello"
    llm = AsyncMock()
    llm.chat.side_effect = RuntimeError("LLM error")
    tts = AsyncMock()

    orch = make_orchestrator(stt, llm, tts, max_retries=0)

    with pytest.raises(PipelineError):
        await orch.run(audio_bytes=b"audio", history=[], session_id="s")

    tts.synthesize.assert_not_called()
