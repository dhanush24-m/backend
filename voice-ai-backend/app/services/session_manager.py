"""
Session State Manager
=====================
Manages per-connection conversation state.

Design
------
- Each WebSocket connection gets a Session object keyed by a UUID session_id.
- No global mutable state: only the SessionManager instance (injected as a
  dependency) holds the registry dict.
- Conversation history is capped at `max_history` turns (configurable) to
  bound memory growth.
- An asyncio background task (`_cleanup_loop`) evicts sessions that have been
  idle longer than `idle_timeout_seconds`.
- Disconnection or timeout both trigger the same `remove_session` path,
  ensuring consistent cleanup.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)
    # OpenAI-style message history [{"role": ..., "content": ...}]
    history: List[dict] = field(default_factory=list)

    def touch(self) -> None:
        self.last_active_at = time.monotonic()

    def add_user_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self.touch()

    def add_assistant_message(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self.touch()

    def truncate_history(self, max_turns: int) -> None:
        """Keep only the last `max_turns` messages."""
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]


class SessionManager:
    def __init__(
        self,
        idle_timeout_seconds: Optional[float] = None,
        max_history: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._idle_timeout = idle_timeout_seconds or settings.session_idle_timeout_seconds
        self._max_history = max_history or settings.max_conversation_history
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start_background_cleanup(self) -> None:
        """Call once after the event loop is running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    # ── Session CRUD ───────────────────────────────────────────────────────────

    async def create_session(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        async with self._lock:
            self._sessions[session_id] = session
        logger.info("Session created", extra={"session_id": session_id})
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def remove_session(self, session_id: str, reason: str = "disconnect") -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            logger.info(
                "Session removed",
                extra={"session_id": session_id, "reason": reason},
            )

    async def add_user_turn(self, session_id: str, text: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.add_user_message(text)
                session.truncate_history(self._max_history)

    async def add_assistant_turn(self, session_id: str, text: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.add_assistant_message(text)
                session.truncate_history(self._max_history)

    async def get_history(self, session_id: str) -> List[dict]:
        async with self._lock:
            session = self._sessions.get(session_id)
            return list(session.history) if session else []

    # ── Background cleanup ─────────────────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        # Check every half-idle-timeout period
        interval = max(10.0, self._idle_timeout / 2)
        while True:
            await asyncio.sleep(interval)
            await self._evict_idle_sessions()

    async def _evict_idle_sessions(self) -> None:
        now = time.monotonic()
        async with self._lock:
            stale = [
                sid
                for sid, s in self._sessions.items()
                if now - s.last_active_at > self._idle_timeout
            ]
            for sid in stale:
                del self._sessions[sid]
        for sid in stale:
            logger.info("Session evicted (idle timeout)", extra={"session_id": sid})
