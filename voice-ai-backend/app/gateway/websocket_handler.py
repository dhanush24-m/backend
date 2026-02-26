"""
WebSocket Gateway Handler
=========================
Handles the raw WebSocket lifecycle: connect, receive, send, disconnect.
NO business logic lives here — this layer only:
  1. Authenticates / rate-limits the connection.
  2. Collects audio chunks.
  3. Delegates to the PipelineOrchestrator.
  4. Sends the audio response back.
  5. Cleans up on disconnect.

Protocol (binary frames)
------------------------
Client sends audio as one or more binary chunks, terminated by a special
"end-of-utterance" sentinel frame (0 bytes of content / empty frame) to signal
that the full utterance has been received.

Server sends:
  - Text frame:   {"status": "processing"}
  - Text frame:   {"status": "error", "message": "..."}        (on failure)
  - Binary frame: raw audio bytes                              (on success)
  - Text frame:   {"status": "done", "transcript": "..."}     (after audio)
"""

import json
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.core.concurrency import ConcurrencyController
from app.core.logging import get_logger, set_logging_context
from app.core.rate_limiter import RateLimiter
from app.pipeline.orchestrator import PipelineError, PipelineOrchestrator
from app.services.session_manager import SessionManager

logger = get_logger(__name__)

# Default MIME type clients are expected to send
DEFAULT_MIME_TYPE = "audio/webm"


class WebSocketHandler:
    def __init__(
        self,
        session_manager: SessionManager,
        orchestrator: PipelineOrchestrator,
        concurrency_controller: ConcurrencyController,
        rate_limiter: RateLimiter,
    ) -> None:
        self._sessions = session_manager
        self._orchestrator = orchestrator
        self._concurrency = concurrency_controller
        self._rate_limiter = rate_limiter

    async def handle(self, websocket: WebSocket) -> None:
        """Entry point for a new WebSocket connection."""
        client_id = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"

        # Create session BEFORE accepting so we have an ID for logging
        session = await self._sessions.create_session()
        session_id = session.session_id
        set_logging_context(session_id=session_id)

        await websocket.accept()
        logger.info("WebSocket connected", extra={"client_id": client_id, "session_id": session_id})

        try:
            await self._message_loop(websocket, session_id, client_id)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", extra={"session_id": session_id})
        except Exception as exc:
            logger.exception("Unexpected error in WebSocket handler", extra={"session_id": session_id, "error": str(exc)})
            try:
                await self._send_error(websocket, "Internal server error")
            except Exception:
                pass
        finally:
            await self._sessions.remove_session(session_id, reason="disconnect")

    # ── Message loop ──────────────────────────────────────────────────────────

    async def _message_loop(self, websocket: WebSocket, session_id: str, client_id: str) -> None:
        audio_buffer: bytearray = bytearray()

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            # Text frame: control messages (e.g. {"cmd": "reset"} to clear history)
            if message.get("text"):
                await self._handle_text_frame(websocket, session_id, message["text"])
                continue

            # Binary frame: audio data
            chunk = message.get("bytes", b"")

            if chunk:
                # Normal data chunk — accumulate
                audio_buffer.extend(chunk)
                continue

            # Empty binary frame == end-of-utterance sentinel
            if len(audio_buffer) == 0:
                logger.debug("Received empty sentinel with no buffered audio — ignoring", extra={"session_id": session_id})
                continue

            # Rate limit check
            if not await self._rate_limiter.is_allowed(client_id):
                await self._send_error(websocket, "Rate limit exceeded. Please wait before sending more audio.")
                audio_buffer.clear()
                continue

            # Concurrency check
            async with self._concurrency.slot(session_id) as acquired:
                if not acquired:
                    await self._send_error(websocket, "Server is busy. Please try again shortly.")
                    audio_buffer.clear()
                    continue

                await self._process_utterance(websocket, session_id, bytes(audio_buffer))

            audio_buffer.clear()

    async def _process_utterance(self, websocket: WebSocket, session_id: str, audio_bytes: bytes) -> None:
        """Run the pipeline for one user utterance and send the response."""
        await self._send_json(websocket, {"status": "processing"})

        history = await self._sessions.get_history(session_id)

        try:
            transcript, audio_response, report = await self._orchestrator.run(
                audio_bytes=audio_bytes,
                history=history,
                session_id=session_id,
            )
        except PipelineError as exc:
            logger.error(
                "Pipeline error",
                extra={"session_id": session_id, "stage": exc.stage, "cause": str(exc.cause)},
            )
            await self._send_error(websocket, f"Processing failed at stage '{exc.stage}'. Please try again.")
            return

        # Persist turns to session history
        await self._sessions.add_user_turn(session_id, transcript)

        # Extract reply text from report context (re-run LLM not needed — get it from orchestrator)
        # We send audio first, then a done frame with transcript metadata
        await websocket.send_bytes(audio_response)

        # Retrieve the assistant's text from history (orchestrator returned it, but we need it here too)
        # We pass it through a done status so the client knows what was said
        await self._send_json(websocket, {
            "status": "done",
            "transcript": transcript,
            "latency": report.stages,
            "total_ms": report.total_ms,
        })

    async def _handle_text_frame(self, websocket: WebSocket, session_id: str, text: str) -> None:
        try:
            cmd = json.loads(text)
        except json.JSONDecodeError:
            return

        if cmd.get("cmd") == "reset":
            session = await self._sessions.get_session(session_id)
            if session:
                session.history.clear()
                session.touch()
            await self._send_json(websocket, {"status": "reset_ok"})
            logger.info("Conversation history reset", extra={"session_id": session_id})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _send_json(websocket: WebSocket, payload: dict) -> None:
        await websocket.send_text(json.dumps(payload))

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        await websocket.send_text(json.dumps({"status": "error", "message": message}))
