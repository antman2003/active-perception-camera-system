"""
Session 30 — Step 6: cumulative voice / ASR / intent / execution counters.

Thread-safe increments from the voice worker (background thread) and executor.
Snapshot dicts are logged to blackbox as ``voice_ptt_cycle_end`` / session final.
"""

from __future__ import annotations

import threading
from typing import Any


class VoiceObsCounters:
    """
    Sparse integer counters (only keys that were bumped appear in ``snapshot()``).
    """

    __slots__ = ("_lock", "_counts")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def add(self, key: str, delta: int = 1) -> None:
        if delta == 0:
            return
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + int(delta)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._counts.items()))


def record_voice_obs(context: Any, **deltas: int) -> None:
    """Increment voice observability counters on ``context.voice_obs`` if present."""
    obs = getattr(context, "voice_obs", None)
    if obs is None or not isinstance(obs, VoiceObsCounters):
        return
    for k, v in deltas.items():
        if v:
            obs.add(k, int(v))
