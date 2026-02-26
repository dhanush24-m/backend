"""
Unit tests for SessionManager.
"""

import asyncio
import time

import pytest

from app.services.session_manager import SessionManager


@pytest.mark.asyncio
async def test_create_session_returns_unique_ids():
    sm = SessionManager(idle_timeout_seconds=60, max_history=10)
    s1 = await sm.create_session()
    s2 = await sm.create_session()
    assert s1.session_id != s2.session_id


@pytest.mark.asyncio
async def test_get_session_returns_created_session():
    sm = SessionManager(idle_timeout_seconds=60, max_history=10)
    session = await sm.create_session()
    retrieved = await sm.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id


@pytest.mark.asyncio
async def test_get_session_returns_none_for_unknown_id():
    sm = SessionManager(idle_timeout_seconds=60, max_history=10)
    result = await sm.get_session("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_add_and_retrieve_history():
    sm = SessionManager(idle_timeout_seconds=60, max_history=10)
    session = await sm.create_session()
    sid = session.session_id

    await sm.add_user_turn(sid, "Hello")
    await sm.add_assistant_turn(sid, "Hi there!")

    history = await sm.get_history(sid)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


@pytest.mark.asyncio
async def test_history_truncated_to_max():
    sm = SessionManager(idle_timeout_seconds=60, max_history=4)
    session = await sm.create_session()
    sid = session.session_id

    for i in range(5):
        await sm.add_user_turn(sid, f"msg {i}")

    history = await sm.get_history(sid)
    assert len(history) == 4
    # Should keep most recent messages
    assert history[-1]["content"] == "msg 4"


@pytest.mark.asyncio
async def test_remove_session_clears_state():
    sm = SessionManager(idle_timeout_seconds=60, max_history=10)
    session = await sm.create_session()
    sid = session.session_id

    await sm.add_user_turn(sid, "hello")
    await sm.remove_session(sid, reason="test")

    retrieved = await sm.get_session(sid)
    assert retrieved is None

    history = await sm.get_history(sid)
    assert history == []


@pytest.mark.asyncio
async def test_idle_sessions_are_evicted():
    # Very short timeout to test eviction
    sm = SessionManager(idle_timeout_seconds=0.05, max_history=10)
    session = await sm.create_session()
    sid = session.session_id

    await asyncio.sleep(0.1)  # Let session go idle
    await sm._evict_idle_sessions()

    result = await sm.get_session(sid)
    assert result is None


@pytest.mark.asyncio
async def test_active_session_not_evicted():
    sm = SessionManager(idle_timeout_seconds=5.0, max_history=10)
    session = await sm.create_session()
    sid = session.session_id

    await sm.add_user_turn(sid, "still active")
    await sm._evict_idle_sessions()

    result = await sm.get_session(sid)
    assert result is not None


@pytest.mark.asyncio
async def test_concurrent_session_creation():
    sm = SessionManager(idle_timeout_seconds=60, max_history=10)

    sessions = await asyncio.gather(*[sm.create_session() for _ in range(20)])
    session_ids = {s.session_id for s in sessions}
    assert len(session_ids) == 20  # All unique
