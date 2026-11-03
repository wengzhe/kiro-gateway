# -*- coding: utf-8 -*-

"""
Liveness for streaming responses.

Kiro can stay silent for minutes: before its first byte (62s measured for a large
tool call) and while it streams reasoning or tool input that the gateway buffers
instead of forwarding. Clients with an idle-read timeout (opencode's chunkTimeout,
300s by default) then abort and resend the whole request.

with_idle_keepalive() wraps the final SSE iterator of a route and writes a
protocol-specific heartbeat whenever the client has received nothing for an
interval. Because heartbeats would otherwise hide a truly stuck upstream forever,
the client must still get an error when Kiro itself stops sending: that limit is
the httpx read timeout (STREAMING_READ_TIMEOUT) on the upstream connection, which
measures raw Kiro bytes. StreamTimings records what happened so a slow upstream
can be told apart from a stuck one in the logs.
"""

import asyncio
import contextlib
import contextvars
import time
from typing import AsyncIterator, Optional

import httpx
from loguru import logger


OPENAI_KEEPALIVE = ": keepalive\n\n"
ANTHROPIC_KEEPALIVE = 'event: ping\ndata: {"type": "ping"}\n\n'

current_timings: contextvars.ContextVar[Optional["StreamTimings"]] = contextvars.ContextVar(
    "current_stream_timings", default=None
)


class StreamTimings:
    """Per-request timing of one streaming response, summarized in a single log line."""

    def __init__(self, label: str):
        self.label = label
        self.started = time.monotonic()
        self.headers_at: Optional[float] = None
        self.upstream_chunks = 0
        self.first_upstream_byte: Optional[float] = None
        self.max_upstream_gap = 0.0
        self.attempts = 0
        self.keepalives = 0
        self._last_upstream: Optional[float] = None

    def upstream_headers_received(self) -> None:
        if self.headers_at is None:
            self.headers_at = time.monotonic() - self.started

    def attempt_started(self) -> None:
        self.attempts += 1
        self._last_upstream = time.monotonic()

    def upstream_chunk(self) -> None:
        now = time.monotonic()
        if self.first_upstream_byte is None:
            self.first_upstream_byte = now - self.started
        if self._last_upstream is not None:
            self.max_upstream_gap = max(self.max_upstream_gap, now - self._last_upstream)
        self._last_upstream = now
        self.upstream_chunks += 1

    def summary(self, outcome: str) -> str:
        fmt = lambda v: f"{v:.1f}s" if v is not None else "none"
        return (
            f"[StreamTimings] {self.label} {outcome}: total={time.monotonic() - self.started:.1f}s "
            f"upstream_headers={fmt(self.headers_at)} first_upstream_byte={fmt(self.first_upstream_byte)} "
            f"max_upstream_gap={self.max_upstream_gap:.1f}s upstream_chunks={self.upstream_chunks} "
            f"attempts={self.attempts} keepalives={self.keepalives}"
        )


async def with_idle_keepalive(
    source: AsyncIterator[str],
    interval: float,
    heartbeat: str,
    timings: Optional[StreamTimings] = None,
) -> AsyncIterator[str]:
    """
    Yields everything from source, plus heartbeat after each interval with no output.

    A single pending read is kept across heartbeats, so source is never read
    concurrently. On close the pending read is cancelled and awaited before the
    source is closed. interval <= 0 passes source through unchanged.
    """
    pending: Optional[asyncio.Task] = None
    try:
        if interval <= 0:
            async for chunk in source:
                yield chunk
            return
        while True:
            if pending is None:
                pending = asyncio.ensure_future(source.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                if timings:
                    timings.keepalives += 1
                yield heartbeat
                continue
            task, pending = pending, None
            try:
                chunk = task.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                await pending
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


async def timed_byte_iterator(response: httpx.Response) -> AsyncIterator[bytes]:
    """Iterates response bytes and records upstream timing for the current request."""
    timings = current_timings.get()
    async for chunk in response.aiter_bytes():
        if timings:
            timings.upstream_chunk()
        yield chunk


def describe_stream_error(error: BaseException, read_timeout: float) -> str:
    """Message for a streaming failure, naming an upstream stall explicitly."""
    if isinstance(error, httpx.ReadTimeout):
        return f"Upstream stalled: Kiro sent no data for {read_timeout:.0f}s"
    return str(error) or type(error).__name__
