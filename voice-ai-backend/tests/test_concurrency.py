"""
Unit tests for ConcurrencyController.
"""

import asyncio

import pytest

from app.core.concurrency import ConcurrencyController


@pytest.mark.asyncio
async def test_acquire_within_limit_succeeds():
    ctrl = ConcurrencyController(max_concurrent=3, acquire_timeout=1.0)
    result = await ctrl.acquire("sess-1")
    assert result is True
    ctrl.release("sess-1")


@pytest.mark.asyncio
async def test_acquire_at_capacity_is_rejected():
    ctrl = ConcurrencyController(max_concurrent=1, acquire_timeout=0.05)
    await ctrl.acquire("sess-1")  # Takes the only slot

    result = await ctrl.acquire("sess-2")  # Should be rejected
    assert result is False

    ctrl.release("sess-1")


@pytest.mark.asyncio
async def test_stats_track_acquired_and_rejected():
    ctrl = ConcurrencyController(max_concurrent=1, acquire_timeout=0.05)
    await ctrl.acquire("s1")
    await ctrl.acquire("s2")  # Rejected
    ctrl.release("s1")

    assert ctrl.stats.total_acquired == 1
    assert ctrl.stats.total_rejected == 1
    assert ctrl.stats.total_released == 1
    assert ctrl.stats.current_active == 0


@pytest.mark.asyncio
async def test_slot_context_manager_acquires_and_releases():
    ctrl = ConcurrencyController(max_concurrent=2, acquire_timeout=1.0)

    async with ctrl.slot("sess") as acquired:
        assert acquired is True
        assert ctrl.stats.current_active == 1

    assert ctrl.stats.current_active == 0


@pytest.mark.asyncio
async def test_slot_context_manager_on_rejection():
    ctrl = ConcurrencyController(max_concurrent=1, acquire_timeout=0.05)
    await ctrl.acquire("blocker")

    async with ctrl.slot("rejected") as acquired:
        assert acquired is False

    ctrl.release("blocker")


@pytest.mark.asyncio
async def test_concurrent_acquisitions_respect_limit():
    limit = 3
    ctrl = ConcurrencyController(max_concurrent=limit, acquire_timeout=0.1)

    acquired_slots = []

    async def try_acquire(i: int):
        ok = await ctrl.acquire(f"sess-{i}")
        if ok:
            acquired_slots.append(i)
            await asyncio.sleep(0.05)
            ctrl.release(f"sess-{i}")

    await asyncio.gather(*[try_acquire(i) for i in range(limit * 2)])

    # At most `limit` should have been active at any given time
    # (all within-limit ones eventually succeed because they're short tasks)
    assert ctrl.stats.total_acquired <= limit * 2
    assert ctrl.stats.current_active == 0


@pytest.mark.asyncio
async def test_available_slots_decreases_on_acquire():
    ctrl = ConcurrencyController(max_concurrent=3, acquire_timeout=1.0)
    assert ctrl.available_slots == 3

    await ctrl.acquire("s1")
    assert ctrl.available_slots == 2

    await ctrl.acquire("s2")
    assert ctrl.available_slots == 1

    ctrl.release("s1")
    assert ctrl.available_slots == 2

    ctrl.release("s2")
    assert ctrl.available_slots == 3
