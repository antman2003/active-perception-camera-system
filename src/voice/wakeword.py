"""
Session 31 — wake word detector abstraction.

Design constraint: in *wake scanning* state we only feed fixed 80 ms chunks
(1280 samples at 16 kHz mono) into the wake model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pathlib import Path


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
            import openwakeword  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Wake word needs the `openwakeword` package.\n"
                "  pip install openwakeword onnxruntime\n"
            ) from e

        self._threshold = float(threshold)
        pkg_dir = Path(str(getattr(openwakeword, "__file__", ""))).resolve().parent
        models_dir = pkg_dir / "resources" / "models"

        def _resolve_model_specs(specs: list[str] | None) -> list[str] | None:
            """
            Allow `--voice-wake-models` to accept:
            - absolute/relative paths to .onnx/.tflite
            - bare model names like `hey_jarvis` / `hey_jarvis_v0.1`
            - custom names like `hey_andrew` (expects a file under resources/models or a path)
            """
            if specs is None:
                return None
            out: list[str] = []
            missing: list[str] = []
            for raw in specs:
                s = str(raw).strip()
                if not s:
                    continue
                p = Path(s)
                if p.is_file():
                    out.append(str(p))
                    continue
                # Try resolving under openwakeword's resources/models.
                candidates: list[Path] = []
                if models_dir:
                    # If user passes a filename with extension, try models_dir/<name>.
                    if p.suffix.lower() in (".onnx", ".tflite"):
                        candidates.append(models_dir / p.name)
                    else:
                        # Common patterns in openWakeWord releases.
                        candidates.append(models_dir / f"{s}.onnx")
                        candidates.append(models_dir / f"{s}_v0.1.onnx")
                        candidates.append(models_dir / f"{s}.tflite")
                        candidates.append(models_dir / f"{s}_v0.1.tflite")
                hit = next((c for c in candidates if c.is_file()), None)
                if hit is not None:
                    out.append(str(hit))
                else:
                    missing.append(s)

            if missing:
                raise FileNotFoundError(
                    "Wake model(s) not found: "
                    + ", ".join(missing)
                    + ". Provide absolute paths to .onnx files, or place them under "
                    + f"{models_dir} (e.g. hey_andrew_v0.1.onnx)."
                )
            return out or None

        resolved_models = _resolve_model_specs(wakeword_models)

        def _build_model(models: list[str] | None) -> Any:
            if models is None:
                # Let openWakeWord select its defaults (built-ins / downloaded models).
                return Model(inference_framework=inference_framework, **kwargs)
            return Model(
                wakeword_models=list(models),
                inference_framework=inference_framework,
                **kwargs,
            )

        try:
            self._model = _build_model(resolved_models)
        except Exception as e:
            # Common Windows failure mode: the pip wheel does not ship model files, so
            # openWakeWord defaults point to resources/models/*.onnx that don't exist.
            # Attempt an automatic download if the user didn't specify explicit models.
            if resolved_models is None:
                try:
                    models_dir.mkdir(parents=True, exist_ok=True)
                    # Download pre-trained models into the expected resources/models directory.
                    # This requires internet on first run, but enables offline wake after.
                    from openwakeword.utils import download_models  # type: ignore

                    download_models()
                    self._model = _build_model(None)
                    return
                except Exception:
                    pass

            # Fall back to a clearer error message.
            raise RuntimeError(
                "openWakeWord model load failed. If you installed openwakeword via pip on Windows, "
                "the wheel may not include model files. Fix:\n"
                "  1) Ensure `requirements-voice.txt` is installed in this venv\n"
                "  2) Run once with internet: python -c \"import openwakeword; openwakeword.utils.download_models()\"\n"
                "  3) Or pass explicit `--voice-wake-models` paths (or names) to existing models\n"
                f"Original error: {e}"
            ) from e

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


class MockWakeWordDetector(WakeWordDetector):
    """
    Deterministic wake detector for CI / demos without microphone or openWakeWord.

    Behavior: every ``period_chunks`` calls, it emits ``streak_len`` **consecutive**
    high-score hits (to satisfy typical ``wake_confirm_chunks`` debouncing), then zeros.
    """

    def __init__(
        self,
        *,
        model_name: str = "mock_wake",
        period_chunks: int = 50,
        streak_len: int = 3,
        score: float = 0.99,
    ) -> None:
        self.model_name = str(model_name)
        self.period_chunks = max(1, int(period_chunks))
        self.streak_len = max(1, int(streak_len))
        self.score = float(score)
        self._i = 0
        self._streak_left = 0

    def predict_chunk(self, pcm_16k_int16: np.ndarray) -> WakeResult:
        self._i += 1
        if self._streak_left > 0:
            self._streak_left -= 1
            s = {self.model_name: self.score}
            return WakeResult(True, self.score, s)
        if (self._i % self.period_chunks) == 0:
            # This chunk + the next (streak_len-1) chunks are hits.
            self._streak_left = self.streak_len - 1
            s = {self.model_name: self.score}
            return WakeResult(True, self.score, s)
        return WakeResult(False, 0.0, {self.model_name: 0.0})

