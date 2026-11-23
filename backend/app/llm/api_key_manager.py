"""
llm/api_key_manager.py
========================
Centralized round-robin manager for the (up to) five Gemini API
keys. Owned exclusively by GeminiClientManager -- agents never touch
this directly.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import GEMINI_API_KEYS, KEY_COOLDOWN_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class _KeyState:
    key: str
    index: int
    cooling_down_until: float = 0.0
    failure_count: int = 0


class NoAvailableKeyError(RuntimeError):
    pass


class APIKeyManager:
    """
    Thread-safe round-robin key rotation with a cooldown for keys
    that just failed (rate limit / quota / auth / transient error),
    so a bad key isn't hammered repeatedly but is also never
    permanently discarded over one failure.
    """

    def __init__(self, keys: List[str] = None, cooldown_seconds: int = KEY_COOLDOWN_SECONDS):
        keys = keys if keys is not None else GEMINI_API_KEYS
        if not keys:
            logger.warning("APIKeyManager initialized with zero API keys.")

        self._states: List[_KeyState] = [
            _KeyState(key=k, index=i) for i, k in enumerate(keys)
        ]
        self._cooldown_seconds = cooldown_seconds
        self._position = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._states)

    def _is_available(self, state: _KeyState, now: float) -> bool:
        return now >= state.cooling_down_until

    def get_next_key(self) -> str:
        """Round-robin: request 1->key1, 2->key2, ..., N->keyN,
        N+1->key1, skipping any key currently in cooldown."""
        with self._lock:
            if not self._states:
                raise NoAvailableKeyError("No Gemini API keys are configured.")

            now = time.monotonic()
            n = len(self._states)

            for offset in range(n):
                candidate_index = (self._position + offset) % n
                state = self._states[candidate_index]
                if self._is_available(state, now):
                    self._position = (candidate_index + 1) % n
                    return state.key

            raise NoAvailableKeyError(
                "All configured Gemini API keys are currently in cooldown."
            )

    def report_failure(self, key: str) -> None:
        """Put `key` into cooldown after a rate limit / quota /
        transient / auth failure. Never permanently discarded."""
        with self._lock:
            for state in self._states:
                if state.key == key:
                    state.failure_count += 1
                    state.cooling_down_until = time.monotonic() + self._cooldown_seconds
                    logger.warning(
                        "Gemini key #%d put into cooldown for %ds (failure #%d).",
                        state.index + 1, self._cooldown_seconds, state.failure_count,
                    )
                    return

    def report_success(self, key: str) -> None:
        """No-op hook kept for symmetry / future metrics; a
        successful call does not need to change key state."""
        return

    def available_key_count(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for s in self._states if self._is_available(s, now))
