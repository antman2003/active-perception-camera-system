"""
Session 31 — wake word detector abstraction.

Design constraint: in *wake scanning* state we only feed fixed 80 ms chunks
(1280 samples at 16 kHz mono) into the wake model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WakeResult:
    detected: bool
    """True when wake threshold is exceeded (per-chunk)."""

    max_score: float
    """Maximum score across all loaded wakeword models for this chunk."""

    scores: dict[str, float]
    """Per-model scores for this chunk."""


class WakeWordDetector:
    """
    Minimal wakeword interface. Implementations should be cheap to call per 80 ms.
    """

    def predict_chunk(self, pcm_16k_int16: np.ndarray) -> WakeResult:  # pragma: no cover
        raise NotImplementedError()


class OpenWakeWordDetector(WakeWordDetector):
    """
    openWakeWord wrapper.

    Notes:
    - openWakeWord expects 16 kHz mono int16 PCM.
    - It maintains an internal rolling buffer (~1.28s) and returns scores per chunk.
    """

    def __init__(
        self,
        *,
        wakeword_models: list[str] | None = None,
        threshold: float = 0.5,
        inference_framework: str = "onnx",
        **kwargs: Any,
    ) -> None:
        try:
            from openwakeword.model import Model  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Wake word needs the `openwakeword` package.\n"
                "  pip install openwakeword onnxruntime\n"
            ) from e

        self._threshold = float(threshold)
        self._model = Model(
            wakeword_models=wakeword_models or None,
            inference_framework=inference_framework,
            **kwargs,
        )

    @property
    def model_names(self) -> list[str]:
        try:
            return list(getattr(self._model, "models", {}).keys())
        except Exception:
            return []

    def predict_chunk(self, pcm_16k_int16: np.ndarray) -> WakeResult:
        x = np.asarray(pcm_16k_int16)
        if x.dtype != np.int16:
            x = x.astype(np.int16)
        x = x.reshape(-1)

        pred = self._model.predict(x)
        # openWakeWord returns a dict {model_name: score} for this chunk.
        scores: dict[str, float] = {}
        if isinstance(pred, dict):
            for k, v in pred.items():
                try:
                    scores[str(k)] = float(v)
                except Exception:
                    continue
        max_score = max(scores.values(), default=0.0)
        return WakeResult(max_score >= self._threshold, max_score, scores)

