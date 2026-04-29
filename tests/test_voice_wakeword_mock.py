"""Session 31 step 4 — mock wake detector behavior."""

from __future__ import annotations

import numpy as np

from src.voice.wakeword import MockWakeWordDetector


def test_mock_wake_single_hit_each_period():
    det = MockWakeWordDetector(period_chunks=3, streak_len=1, score=0.9)
    x = np.zeros(1280, dtype=np.int16)
    r1 = det.predict_chunk(x)
    r2 = det.predict_chunk(x)
    r3 = det.predict_chunk(x)
    assert r1.detected is False
    assert r2.detected is False
    assert r3.detected is True
    assert r3.max_score == 0.9


def test_mock_wake_streak_satisfies_confirm_window():
    det = MockWakeWordDetector(period_chunks=10, streak_len=3, score=0.9)
    x = np.zeros(1280, dtype=np.int16)
    for _ in range(9):
        assert det.predict_chunk(x).detected is False
    for _ in range(3):
        r = det.predict_chunk(x)
        assert r.detected is True
        assert r.max_score == 0.9
    assert det.predict_chunk(x).detected is False
