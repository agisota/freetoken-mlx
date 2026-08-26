"""Cancellation-path tests for FrontendManager.stream_with_cancellation / abort_user.

Pins the abort contract for a client disconnect mid-stream:

  1. Cancellation performs the cleanup INLINE (shielded await) — exactly one AbortMsg is
     sent, both ack/event maps are emptied, and no orphaned background task is left behind.
  2. A failure while delivering the abort must not swallow the original CancelledError.
  3. abort_user is idempotent: a second call for an already-aborted uid sends nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from freetoken.message import AbortMsg
from freetoken.server.api_server import FrontendManager


class _Stats:
    def __init__(self):
        self.aborts = []

    def on_abort(self, uid):
        self.aborts.append(uid)


def _state(send_impl=None):
    """Minimal stand-in exposing only what stream_with_cancellation/abort_user touch."""
    st = SimpleNamespace(
        ack_map={7: [object()]},
        event_map={7: asyncio.Event()},
        stats=_Stats(),
    )
    sent = []

    async def default_send(msg):
        sent.append(msg)

    st.send_one = send_impl or default_send
    st.sent = sent
    return st


class _Request:
    def __init__(self, disconnected=False):
        self._disconnected = disconnected

    async def is_disconnected(self):
        return self._disconnected


async def _consume(state, request, uid=7):
    state.abort_user = lambda request_uid: FrontendManager.abort_user(state, request_uid)
    async for _ in FrontendManager.stream_with_cancellation(state, _never(), request, uid):
        pass


async def _never():
    await asyncio.sleep(3600)
    yield b""  # pragma: no cover - unreachable


def test_cancellation_sends_one_abort_and_cleans_maps_inline():
    async def run():
        state = _state()
        task = asyncio.create_task(_consume(state, _Request()))
        await asyncio.sleep(0.01)  # let the consumer suspend inside the generator
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Inline awaited cleanup: nothing but ourselves may remain.
        assert asyncio.all_tasks() - {asyncio.current_task()} == set()
        assert len(state.sent) == 1
        assert isinstance(state.sent[0], AbortMsg)
        assert state.sent[0].uid == 7
        assert state.ack_map == {} and state.event_map == {}
        assert state.stats.aborts == [7]

    asyncio.run(run())


def test_abort_delivery_failure_preserves_cancellation():
    async def boom(msg):  # abort delivery fails at the ZMQ layer
        raise RuntimeError("zmq down")

    async def run():
        state = _state(send_impl=boom)
        task = asyncio.create_task(_consume(state, _Request()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task  # the delivery failure must NOT replace the cancellation

    asyncio.run(run())


def test_abort_user_is_idempotent():
    async def run():
        state = _state()
        await FrontendManager.abort_user(state, 7)
        assert len(state.sent) == 1
        # Maps already empty -> second call is a no-op, no duplicate AbortMsg.
        await FrontendManager.abort_user(state, 7)
        assert len(state.sent) == 1

    asyncio.run(run())
