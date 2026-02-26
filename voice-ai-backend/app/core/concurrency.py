"""
Concurrency Controller
======================
Limits the number of active AI pipeline executions per server instance.

Design
------
- Uses asyncio.Semaphore so that acquiring a slot is non-blocking for the caller
  (it awaits rather than blocking the event loop).
- If the semaphore cannot be acquired within `acquire_timeout` seconds the
  request is rejected with a clear error — this is the backpressure strategy.
- The semaphore count is configurable via environment (max_concurrent_pipelines).

Trade-offs
----------
- In-process semaphore works for a single instance only.  For multi-instance
  deployments a distributed counter (e.g. Redis INCR/DECR) would be needed.
- Rejection is chosen over queuing to keep latency predictable; callers can
  retry using their own back-off.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConcurrencyStats:
    total_acquired: int = 0
    total_rejected: int = 0
    total_released: int = 0
    current_active: int = 0


class ConcurrencyController:
    def __init__(self, max_concurrent: int, acquire_timeout: float = 2.0) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent
        self._acquire_timeout = acquire_timeout
        self._stats = ConcurrencyStats()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def acquire(self, session_id: str = "") -> bool:
        """
        Try to acquire a pipeline slot.

        Returns True if acquired; False if the system is at capacity.
        """
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._acquire_timeout,
            )
        except asyncio.TimeoutError:
            self._stats.total_rejected += 1
            logger.warning(
                "Pipeline slot rejected — system at capacity",
                extra={"session_id": session_id, "max_concurrent": self._max},
            )
            return False

        self._stats.total_acquired += 1
        self._stats.current_active += 1
        logger.debug(
            "Pipeline slot acquired",
            extra={
                "session_id": session_id,
                "current_active": self._stats.current_active,
                "capacity": self._max,
            },
        )
        return True

    def release(self, session_id: str = "") -> None:
        """Release a previously acquired slot."""
        self._semaphore.release()
        self._stats.total_released += 1
        self._stats.current_active = max(0, self._stats.current_active - 1)
        logger.debug(
            "Pipeline slot released",
            extra={
                "session_id": session_id,
                "current_active": self._stats.current_active,
            },
        )

    @property
    def stats(self) -> ConcurrencyStats:
        return self._stats

    @property
    def available_slots(self) -> int:
        return self._semaphore._value  # type: ignore[attr-defined]

    # ── Async context manager ──────────────────────────────────────────────────

    class _Slot:
        def __init__(self, controller: "ConcurrencyController", session_id: str):
            self._ctrl = controller
            self._session_id = session_id
            self._acquired = False

        async def __aenter__(self) -> bool:
            self._acquired = await self._ctrl.acquire(self._session_id)
            return self._acquired

        async def __aexit__(self, *_: object) -> None:
            if self._acquired:
                self._ctrl.release(self._session_id)

    def slot(self, session_id: str = "") -> "_Slot":
        """
        Usage::

            async with controller.slot(session_id) as acquired:
                if not acquired:
                    # handle rejection
                    return
                # run pipeline
        """
        return self._Slot(self, session_id)
