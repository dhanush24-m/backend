"""
Rate Limiter
============
Per-client, in-memory sliding-window rate limiting.

Trade-offs
----------
- Stored in a dict keyed by client_id (e.g. remote address or session ID).
- Uses a deque of request timestamps; entries older than the window are
  purged on every check — O(n) per check, acceptable for typical usage.
- In-memory only: does not survive restarts and does not share state across
  multiple processes/hosts.  For distributed deployments use Redis + Lua.
"""

import asyncio
import time
from collections import deque
from typing import Dict

from app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        # client_id -> deque of epoch timestamps (float)
        self._buckets: Dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        """Return True if the client is within its rate limit."""
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets.setdefault(client_id, deque())

            # Evict expired entries
            cutoff = now - self._window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self._max:
                logger.warning(
                    "Rate limit exceeded",
                    extra={"client_id": client_id, "limit": self._max, "window": self._window},
                )
                return False

            bucket.append(now)
            return True

    async def cleanup_stale(self) -> None:
        """Remove clients with no recent activity (call periodically)."""
        now = time.monotonic()
        async with self._lock:
            stale = [
                cid
                for cid, bucket in self._buckets.items()
                if not bucket or bucket[-1] < now - self._window
            ]
            for cid in stale:
                del self._buckets[cid]
        if stale:
            logger.debug("Rate limiter cleanup", extra={"removed_clients": len(stale)})
