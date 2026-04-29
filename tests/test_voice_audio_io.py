"""Session 31 step 3 — audio IO rebuffering tests (no microphone required)."""

from __future__ import annotations

import numpy as np

from src.voice.audio_io import ChunkRebuffer, chunk_samples


def test_chunk_samples_16k_80ms_is_1280():
    assert chunk_samples(16000, 80) == 1280


def test_rebuffer_exact_chunks():
    reb = ChunkRebuffer(4)
    out = reb.push(np.array([1, 2, 3, 4], dtype=np.int16))
    assert len(out) == 1
    assert out[0].tolist() == [1, 2, 3, 4]


def test_rebuffer_smaller_blocks_accumulate():
    reb = ChunkRebuffer(4)
    assert reb.push(np.array([1, 2], dtype=np.int16)) == []
    out = reb.push(np.array([3, 4], dtype=np.int16))
    assert len(out) == 1
    assert out[0].tolist() == [1, 2, 3, 4]


def test_rebuffer_larger_blocks_split():
    reb = ChunkRebuffer(4)
    out = reb.push(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16))
    assert [c.tolist() for c in out] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    # remaining 9 buffered; pushing 3 more yields the last chunk
    out2 = reb.push(np.array([10, 11, 12], dtype=np.int16))
    assert len(out2) == 1
    assert out2[0].tolist() == [9, 10, 11, 12]

