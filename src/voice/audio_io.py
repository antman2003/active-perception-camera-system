"""
Session 31 — Step 3: audio capture standardization.

Goal: provide a stable contract to downstream modules:

- Sample rate: 16 kHz
- Channels: mono
- dtype: int16
- Chunk size: 80 ms = 1280 samples (@16k)

Windows / PortAudio stacks can return different block sizes or float32 by default.
This module absorbs those differences and yields fixed-size int16 chunks.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np


def chunk_samples(sample_rate_hz: int = 16000, chunk_ms: int = 80) -> int:
    return max(1, int(round(int(sample_rate_hz) * (float(chunk_ms) / 1000.0))))


class ChunkRebuffer:
    """
    Rebuffer arbitrary int16 audio blocks into fixed-size int16 chunks.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = max(1, int(chunk_size))
        self._buf = np.zeros((0,), dtype=np.int16)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(block)
        if x.dtype != np.int16:
            x = x.astype(np.int16)
        x = x.reshape(-1)
        if x.size == 0:
            return []
        self._buf = np.concatenate([self._buf, x], axis=0)
        out: list[np.ndarray] = []
        while self._buf.size >= self.chunk_size:
            out.append(self._buf[: self.chunk_size].copy())
            self._buf = self._buf[self.chunk_size :]
        return out


@dataclass(frozen=True)
class MicStreamConfig:
    sample_rate_hz: int = 16000
    chunk_ms: int = 80
    device: str | int | None = None
    q_max_chunks: int = 64


class MicChunkStream:
    """
    Context manager that yields fixed-size int16 chunks from the default input device.

    Internally uses `sounddevice.InputStream` with dtype int16 when possible.
    If PortAudio returns unexpected block sizes, we rebuffer them to our chunk size.
    """

    def __init__(self, cfg: MicStreamConfig):
        self.cfg = cfg
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max(4, int(cfg.q_max_chunks)))
        self._stop = threading.Event()
        self._stream = None
        self._reb = ChunkRebuffer(chunk_samples(cfg.sample_rate_hz, cfg.chunk_ms))
        self._last_status: str | None = None

    @property
    def last_status(self) -> str | None:
        return self._last_status

    def consume_status(self) -> str | None:
        """Return and clear the last PortAudio status string (if any)."""
        s = self._last_status
        self._last_status = None
        return s

    def __enter__(self) -> "MicChunkStream":
        import sounddevice as sd

        sr = int(self.cfg.sample_rate_hz)
        # We *request* our ideal blocksize, but PortAudio may still vary.
        req_blocksize = chunk_samples(sr, int(self.cfg.chunk_ms))

        def _cb(indata, frames, _t, status) -> None:
            if status:
                self._last_status = str(status)
            if self._stop.is_set():
                return

            x = np.asarray(indata)
            # sounddevice may deliver shape (frames, channels); enforce mono.
            if x.ndim == 2:
                if x.shape[1] > 1:
                    # int16 stereo: average channels in int32 then cast back.
                    x = (x.astype(np.int32).mean(axis=1)).astype(np.int16)
                else:
                    x = x[:, 0]
            x = x.reshape(-1)

            for c in self._reb.push(x):
                try:
                    self._q.put_nowait(c)
                except queue.Full:
                    # Drop oldest to keep latency bounded.
                    try:
                        _ = self._q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(c)
                    except queue.Full:
                        pass

        self._stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=req_blocksize,
            callback=_cb,
            device=self.cfg.device,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None

    def chunks(self, *, timeout_s: float = 0.25) -> Iterator[np.ndarray]:
        """
        Yield chunks until the context manager is closed.
        """
        while not self._stop.is_set():
            try:
                yield self._q.get(timeout=timeout_s)
            except queue.Empty:
                continue


def list_input_devices() -> list[dict]:
    """
    Convenience helper for debugging: returns PortAudio device dicts that have input channels.
    """
    import sounddevice as sd

    out: list[dict] = []
    for i, d in enumerate(sd.query_devices()):
        try:
            if int(d.get("max_input_channels", 0)) > 0:
                out.append({"index": i, **dict(d)})
        except Exception:
            continue
    return out

