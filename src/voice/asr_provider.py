"""
Session 30 — Step 2: ASR behind a small provider interface.

- ``FasterWhisperAsrProvider``: local **faster-whisper** + OpenAI **Whisper** weights
  (default ``model_size="base"``; first run downloads into the Hugging Face Hub cache).
- ``MockAsrProvider``: fixed text for CI / no-GPU environments.

Verification checklist: ``docs/VOICE_ASR.md`` §3.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Union

import numpy as np

# Public contract: path to audio file, or mono PCM at ``sample_rate_hz`` when using ndarray.
WavOrPcm = Union[str, Path, np.ndarray, BinaryIO]


@dataclass
class AsrResult:
    """Outcome of one ``transcribe`` call."""

    text: str
    """Full transcript (segments concatenated)."""

    segments: list[dict[str, Any]] = field(default_factory=list)
    """Lightweight segment records: start, end, text."""

    t_asr_ms: int = 0
    """Wall time spent inside the ASR engine for this call."""

    language: str | None = None
    """Detected or forced language code, when known."""

    duration_audio_s: float | None = None
    """Input audio duration in seconds, when known."""


class AsrProvider(ABC):
    """Speech → text. Implementations must not touch serial / pan-tilt."""

    @abstractmethod
    def transcribe(
        self,
        wav_or_pcm: WavOrPcm,
        *,
        lang_hint: str | None = None,
        sample_rate_hz: int | None = None,
    ) -> AsrResult:
        """
        :param wav_or_pcm: Path-like to a sound file, **or** mono ``float32`` / ``int16``
            ``np.ndarray`` (use ``sample_rate_hz``; will be converted to 16 kHz float mono).
        :param lang_hint: e.g. ``\"zh\"``, ``\"en\"``, or ``\"auto\"`` / ``None`` for detector.
        :param sample_rate_hz: Required when ``wav_or_pcm`` is an ndarray.
        """


class MockAsrProvider(AsrProvider):
    """Returns a constant transcript; ignores audio input."""

    def __init__(self, text: str = "回家"):
        self._text = text

    def transcribe(
        self,
        wav_or_pcm: WavOrPcm,
        *,
        lang_hint: str | None = None,
        sample_rate_hz: int | None = None,
    ) -> AsrResult:
        return AsrResult(
            text=self._text,
            segments=[{"start": 0.0, "end": 0.0, "text": self._text}],
            t_asr_ms=0,
            language=lang_hint if lang_hint not in (None, "", "auto") else None,
            duration_audio_s=None,
        )


def _linear_resample_mono(x: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Cheap linear resample (good enough for ASR front-end)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if orig_sr == target_sr:
        return x.astype(np.float32)
    duration_s = len(x) / float(orig_sr)
    n_new = max(1, int(round(duration_s * target_sr)))
    t_old = np.linspace(0.0, duration_s, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, duration_s, num=n_new, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def _numpy_audio_to_float_mono_16k(
    wav_or_pcm: np.ndarray, sample_rate_hz: int
) -> np.ndarray:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    x = np.asarray(wav_or_pcm)
    if x.ndim == 2:
        x = x.mean(axis=1)
    elif x.ndim != 1:
        raise ValueError("audio ndarray must be 1-D mono or 2-D (frames, channels)")
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32)
    return _linear_resample_mono(x, sample_rate_hz, 16000)


class FasterWhisperAsrProvider(AsrProvider):
    """
    faster-whisper backend. First ``transcribe`` may download / compile weights — can
    take tens of seconds on a cold machine.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._vad_filter = vad_filter
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from e
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._model

    def transcribe(
        self,
        wav_or_pcm: WavOrPcm,
        *,
        lang_hint: str | None = None,
        sample_rate_hz: int | None = None,
    ) -> AsrResult:
        model = self._ensure_model()
        language = None
        if lang_hint and str(lang_hint).lower() not in ("auto", ""):
            language = str(lang_hint)

        t0 = time.perf_counter()
        audio_arg: Union[str, np.ndarray, Path]
        duration_s: float | None = None

        if isinstance(wav_or_pcm, (str, Path)):
            audio_arg = str(Path(wav_or_pcm))
        elif isinstance(wav_or_pcm, np.ndarray):
            if sample_rate_hz is None:
                raise ValueError("sample_rate_hz is required when wav_or_pcm is ndarray")
            audio_arg = _numpy_audio_to_float_mono_16k(wav_or_pcm, int(sample_rate_hz))
            duration_s = float(len(audio_arg)) / 16000.0
        elif hasattr(wav_or_pcm, "read"):
            raise TypeError("BinaryIO is not supported yet; write to a temp .wav or pass ndarray")
        else:
            raise TypeError(f"unsupported audio type: {type(wav_or_pcm)!r}")

        # Real-time-ish defaults: greedy decode is much faster than beam search on CPU.
        # Users can still pick a larger model for quality; this keeps latency usable.
        segments_iter, info = model.transcribe(
            audio_arg,
            beam_size=1,
            best_of=1,
            vad_filter=self._vad_filter,
            language=language,
        )

        segs: list[dict[str, Any]] = []
        parts: list[str] = []
        for seg in segments_iter:
            parts.append(seg.text)
            segs.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text,
                }
            )

        text = "".join(parts).strip()
        t1 = time.perf_counter()
        t_ms = int(round((t1 - t0) * 1000))

        if duration_s is None and getattr(info, "duration", None) is not None:
            duration_s = float(info.duration)

        lang_out = getattr(info, "language", None) or language

        return AsrResult(
            text=text,
            segments=segs,
            t_asr_ms=t_ms,
            language=lang_out,
            duration_audio_s=duration_s,
        )
