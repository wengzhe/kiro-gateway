"""
Session continuation state for Kiro's agentContinuationId / conversationId.

Official kiro-cli reuses one agentContinuationId (and conversationId) for every
upstream call belonging to the same agent task, and only generates new ones for a
new session. Verified by capturing real kiro-cli traffic: two tool-use round trips
shared `bfaf422e-...`, while a fresh session produced `2a9bc885-...`.

Clients identify their session via a session header. opencode 1.18.34+ sends the
namespaced `x-opencode-session-id` alongside the legacy `x-session-id` and
`x-session-affinity` (all carrying the same value, verified on opencode 1.19.0).
The namespaced header is read first so continuation keeps working if the legacy
ones are ever dropped. Requests without any of them fall back to per-request
identifiers, preserving the previous stateless behaviour.
"""

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from kiro.config import (
    SESSION_CONTINUATION_ENABLED,
    SESSION_CONTINUATION_MAX_ENTRIES,
    SESSION_CONTINUATION_TTL_SECONDS,
)

SESSION_ID_HEADERS = ("x-opencode-session-id", "x-session-id", "x-session-affinity")


@dataclass
class ContinuationState:
    conversation_id: str
    agent_continuation_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    request_count: int = 0


class SessionContinuationStore:
    def __init__(
        self,
        max_entries: int = SESSION_CONTINUATION_MAX_ENTRIES,
        ttl_seconds: float = SESSION_CONTINUATION_TTL_SECONDS,
    ):
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, ContinuationState]" = OrderedDict()
        self._lock = threading.Lock()

    def resolve(self, session_id: Optional[str]) -> ContinuationState:
        if not session_id:
            return self._new_state()

        with self._lock:
            self._evict_expired()
            state = self._entries.get(session_id)
            if state is None:
                state = self._new_state()
                self._entries[session_id] = state
                self._evict_overflow()
            else:
                self._entries.move_to_end(session_id)
            state.last_used_at = time.monotonic()
            state.request_count += 1
            return state

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._entries.pop(session_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "tracked_sessions": len(self._entries),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
            }

    def _new_state(self) -> ContinuationState:
        return ContinuationState(
            conversation_id=str(uuid.uuid4()),
            agent_continuation_id=str(uuid.uuid4()),
        )

    def _evict_expired(self) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff = time.monotonic() - self._ttl_seconds
        expired = [key for key, state in self._entries.items() if state.last_used_at < cutoff]
        for key in expired:
            del self._entries[key]

    def _evict_overflow(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


_store = SessionContinuationStore()


def extract_session_id(headers) -> Optional[str]:
    if not SESSION_CONTINUATION_ENABLED:
        return None
    for name in SESSION_ID_HEADERS:
        value = headers.get(name)
        if value:
            return value.strip() or None
    return None


def resolve_continuation(session_id: Optional[str]) -> ContinuationState:
    return _store.resolve(session_id)


def continuation_stats() -> dict:
    return _store.stats()
