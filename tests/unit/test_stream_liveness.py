# -*- coding: utf-8 -*-

"""Tests for kiro.stream_liveness: idle keepalives, stall messages and timings."""

import asyncio

import httpx
import pytest

from kiro.stream_liveness import (
    ANTHROPIC_KEEPALIVE,
    OPENAI_KEEPALIVE,
    StreamTimings,
    current_timings,
    describe_stream_error,
    timed_byte_iterator,
    with_idle_keepalive,
)


async def _collect(agen, until=None):
    out = []
    async with asyncio.timeout(5):
        async for chunk in agen:
            out.append(chunk)
            if until:
                until(chunk)
    return out


class TestWithIdleKeepalive:
    """Tests for with_idle_keepalive()."""

    @pytest.mark.asyncio
    async def test_heartbeat_before_first_output(self):
        """
        What it does: Blocks the source before its first chunk.
        Purpose: The silence before Kiro's first byte must be covered too.
        """
        release = asyncio.Event()
        beats = 0

        async def source():
            await release.wait()
            yield "data: a\n\n"

        def on_chunk(chunk):
            nonlocal beats
            if chunk == OPENAI_KEEPALIVE:
                beats += 1
                if beats == 3:
                    release.set()

        out = await _collect(with_idle_keepalive(source(), 0.01, OPENAI_KEEPALIVE), on_chunk)
        print(f"Output: {out}")
        assert out[:3] == [OPENAI_KEEPALIVE] * 3
        assert out[-1] == "data: a\n\n"

    @pytest.mark.asyncio
    async def test_payload_unchanged_and_ordered(self):
        """
        What it does: Interleaves pauses with real chunks.
        Purpose: Heartbeats must never alter or reorder payload chunks.
        """
        gates = [asyncio.Event(), asyncio.Event()]
        payload = ["data: 1\n\n", "data: 2\n\n", "data: 3\n\n"]

        async def source():
            yield payload[0]
            await gates[0].wait()
            yield payload[1]
            await gates[1].wait()
            yield payload[2]

        opened = 0

        def on_chunk(chunk):
            nonlocal opened
            if chunk == ANTHROPIC_KEEPALIVE and opened < 2:
                gates[opened].set()
                opened += 1

        out = await _collect(with_idle_keepalive(source(), 0.01, ANTHROPIC_KEEPALIVE), on_chunk)
        assert [c for c in out if c != ANTHROPIC_KEEPALIVE] == payload
        assert ANTHROPIC_KEEPALIVE in out

    @pytest.mark.asyncio
    async def test_no_heartbeat_when_source_is_fast(self):
        """
        What it does: Streams a source that never pauses, with a long interval.
        Purpose: Active streams carry no extra frames.
        """
        async def source():
            for i in range(5):
                yield f"data: {i}\n\n"

        out = await _collect(with_idle_keepalive(source(), 60, OPENAI_KEEPALIVE))
        assert OPENAI_KEEPALIVE not in out
        assert len(out) == 5

    @pytest.mark.asyncio
    async def test_zero_interval_passes_through(self):
        """
        What it does: Uses interval=0.
        Purpose: STREAM_KEEPALIVE_INTERVAL=0 disables heartbeats entirely.
        """
        async def source():
            yield "x"

        assert await _collect(with_idle_keepalive(source(), 0, OPENAI_KEEPALIVE)) == ["x"]

    @pytest.mark.asyncio
    async def test_source_error_propagates(self):
        """
        What it does: Lets the source raise after a heartbeat.
        Purpose: Upstream errors (e.g. httpx.ReadTimeout) must reach the route's error handler.
        """
        release = asyncio.Event()

        async def source():
            await release.wait()
            raise httpx.ReadTimeout("stalled")
            yield

        with pytest.raises(httpx.ReadTimeout):
            await _collect(
                with_idle_keepalive(source(), 0.01, OPENAI_KEEPALIVE),
                lambda c: release.set(),
            )

    @pytest.mark.asyncio
    async def test_close_cancels_pending_read_and_closes_source(self):
        """
        What it does: Closes the wrapper while a source read is pending.
        Purpose: A client disconnect must stop the upstream read and run its cleanup.
        """
        read_cancelled = asyncio.Event()
        source_closed = asyncio.Event()

        async def source():
            try:
                yield "first"
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    read_cancelled.set()
                    raise
            finally:
                source_closed.set()

        wrapper = with_idle_keepalive(source(), 0.01, OPENAI_KEEPALIVE)
        async with asyncio.timeout(5):
            async for chunk in wrapper:
                if chunk == OPENAI_KEEPALIVE:
                    break
            await wrapper.aclose()

        assert read_cancelled.is_set()
        assert source_closed.is_set()

    @pytest.mark.asyncio
    async def test_zero_interval_close_still_closes_source(self):
        """
        What it does: Closes a pass-through (interval=0) wrapper before the source ends.
        Purpose: Disabling keepalives must not leak the upstream generator on disconnect.
        """
        source_closed = asyncio.Event()

        async def source():
            try:
                yield "first"
                yield "second"
            finally:
                source_closed.set()

        wrapper = with_idle_keepalive(source(), 0, OPENAI_KEEPALIVE)
        assert await wrapper.__anext__() == "first"
        await wrapper.aclose()

        assert source_closed.is_set()

    @pytest.mark.asyncio
    async def test_counts_keepalives_in_timings(self):
        """
        What it does: Passes a StreamTimings and emits two heartbeats.
        Purpose: The per-request log line reports how many heartbeats were needed.
        """
        release = asyncio.Event()
        timings = StreamTimings("test")

        async def source():
            await release.wait()
            yield "x"

        def on_chunk(chunk):
            if timings.keepalives == 2:
                release.set()

        await _collect(with_idle_keepalive(source(), 0.01, OPENAI_KEEPALIVE, timings), on_chunk)
        assert timings.keepalives == 2


class TestStreamTimings:
    """Tests for StreamTimings and timed_byte_iterator()."""

    @pytest.mark.asyncio
    async def test_records_upstream_chunks_for_current_request(self):
        """
        What it does: Reads a fake response while timings are bound to the context.
        Purpose: The log line must reflect raw Kiro bytes, including ones never forwarded.
        """
        class FakeResponse:
            async def aiter_bytes(self):
                yield b"a"
                yield b"b"

        timings = StreamTimings("POST /v1/chat/completions")
        timings.attempt_started()
        token = current_timings.set(timings)
        try:
            chunks = [c async for c in timed_byte_iterator(FakeResponse())]
        finally:
            current_timings.reset(token)

        line = timings.summary("completed")
        print(line)
        assert chunks == [b"a", b"b"]
        assert timings.upstream_chunks == 2
        assert timings.first_upstream_byte is not None
        assert "completed" in line and "upstream_chunks=2" in line and "attempts=1" in line

    def test_summary_without_upstream_bytes(self):
        """
        What it does: Summarizes a request that never received a byte.
        Purpose: A stalled-before-first-byte request must be recognisable in logs.
        """
        line = StreamTimings("POST /v1/messages").summary("error")
        assert "upstream_headers=none" in line
        assert "first_upstream_byte=none" in line

    def test_records_header_wait(self):
        """
        What it does: Marks upstream headers as received.
        Purpose: Kiro can withhold HTTP headers for over a minute; the log must show it.
        """
        timings = StreamTimings("POST /v1/chat/completions")
        timings.upstream_headers_received()
        assert timings.headers_at is not None
        assert "upstream_headers=none" not in timings.summary("completed")


class TestDescribeStreamError:
    """Tests for describe_stream_error()."""

    def test_read_timeout_names_the_stall(self):
        """
        What it does: Describes an httpx.ReadTimeout.
        Purpose: Clients and logs should say the upstream stalled, not just "ReadTimeout".
        """
        assert describe_stream_error(httpx.ReadTimeout(""), 300) == "Upstream stalled: Kiro sent no data for 300s"

    def test_other_errors_keep_their_message(self):
        assert describe_stream_error(ValueError("boom"), 300) == "boom"
        assert describe_stream_error(RuntimeError(), 300) == "RuntimeError"
