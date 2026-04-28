"""Session 30 step 2 — ASR provider interface."""

import numpy as np
import pytest

from src.voice.asr_provider import (
    AsrResult,
    FasterWhisperAsrProvider,
    MockAsrProvider,
)


def test_mock_transcribe_returns_fixed_text():
    p = MockAsrProvider(text="左转十度")
    r = p.transcribe("ignored", lang_hint="zh")
    assert r.text == "左转十度"
    assert r.t_asr_ms == 0
    assert len(r.segments) == 1


def test_mock_accepts_ndarray_without_sample_rate_for_mock():
    p = MockAsrProvider()
    r = p.transcribe(np.zeros(160, dtype=np.float32))
    assert isinstance(r, AsrResult)


def test_faster_provider_import_guard():
    """Smoke: class is constructible; heavy model loads on first transcribe."""
    fw = FasterWhisperAsrProvider(model_size="tiny", device="cpu", compute_type="int8")
    assert fw._model is None


@pytest.mark.slow
def test_faster_whisper_tiny_on_silence():
    """Optional: downloads weights; skip in CI with ``pytest -m 'not slow'``."""
    prov = FasterWhisperAsrProvider(model_size="tiny", device="cpu", compute_type="int8")
    silence = np.zeros(8000, dtype=np.float32)
    r = prov.transcribe(silence, lang_hint="en", sample_rate_hz=16000)
    assert isinstance(r.text, str)
    assert r.t_asr_ms >= 0
