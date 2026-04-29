"""Session 31 step 6: utterance end metadata (shared with always_worker)."""

from src.voice.always_worker import utterance_capture_end_meta


def test_silence_end_not_truncated():
    r, t = utterance_capture_end_meta(cap_silence_streak=10, hang_chunks=5)
    assert r == "silence"
    assert t is False


def test_max_len_truncated():
    r, t = utterance_capture_end_meta(cap_silence_streak=0, hang_chunks=5)
    assert r == "max_len"
    assert t is True


def test_when_both_silence_and_max_prefer_silence_label():
    r, t = utterance_capture_end_meta(cap_silence_streak=10, hang_chunks=5)
    assert r == "silence"
    assert t is False
