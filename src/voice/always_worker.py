"""
Session 31 — always-on wake word + capture + ASR pipeline.

Data flow:
- Wake scanning: every 80 ms (1280 @16k) chunk goes into wake model only.
- After wake confirmed: capture a full command window (min duration + silence hangover + max cap),
  then send the *whole* captured audio to ASR once.
- After ASR + intent + executor: return to wake scanning with a short refractory cooldown.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.voice.asr_provider import AsrProvider, FasterWhisperAsrProvider
from src.voice.clarify import VoiceIntentClarifySession
from src.voice.executor import execute_voice_commands
from src.voice.intent import _intent_from_rules, _intent_has_non_noop, normalize_zh_command_text
from src.voice.metrics import record_voice_obs
from src.voice.llm_client import OllamaIntentClient
from src.voice.worker import VoiceLlmLoggingClient
from src.voice.wakeword import OpenWakeWordDetector, WakeWordDetector


@dataclass
class AlwaysOnVoiceConfig:
    # Audio IO
    sample_rate_hz: int = 16000
    wake_chunk_ms: int = 80  # must match openWakeWord best-efficiency chunk
    device: str | int | None = None

    # Wake word
    wake_threshold: float = 0.5
    wake_confirm_chunks: int = 3
    wake_refractory_ms: int = 1200
    wake_inference_framework: str = "onnx"
    wake_models: list[str] | None = None

    # Capture window
    capture_min_s: float = 0.8
    capture_max_s: float = 6.0
    capture_silence_hangover_ms: int = 700
    vad_rms_threshold: float = 0.008  # tuned for float32 mic streams

    # Downstream (reuse Session 30 stack)
    lang_hint: str | None = "zh"
    model_size: str = "base"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    use_llm: bool = True
    llm_model: str = "qwen2.5:1.5b"
    enable_clarify: bool = False
    llm_only_on_asr_low_confidence: bool = False


class AlwaysOnVoiceWorker:
    """
    Runs a microphone stream continuously in a background thread.
    """

    def __init__(self, context: Any, cfg: AlwaysOnVoiceConfig):
        self._context = context
        self._cfg = cfg

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VoiceAlwaysOn", daemon=True)

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

        # Wake model
        self._wake: WakeWordDetector = OpenWakeWordDetector(
            wakeword_models=cfg.wake_models,
            threshold=cfg.wake_threshold,
            inference_framework=cfg.wake_inference_framework,
        )

        # ASR + intent plumbing (same as PTT worker)
        self._asr: AsrProvider = FasterWhisperAsrProvider(
            model_size=cfg.model_size,
            device=cfg.asr_device,
            compute_type=cfg.asr_compute_type,
            vad_filter=True,
        )
        self._clarify = VoiceIntentClarifySession()
        self._llm_client = None
        self._llm_textfix_client = None
        self._llm_bundle_client = None
        if cfg.use_llm:
            base = OllamaIntentClient(model=cfg.llm_model)
            wrapped = VoiceLlmLoggingClient(base, context, _StatusShim(self))
            self._llm_client = wrapped
            self._llm_textfix_client = wrapped
            self._llm_bundle_client = wrapped

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)

    # --- HUD / blackbox helpers (match PTT names where possible) ---
    def _set_status(self, status: str) -> None:
        try:
            setattr(self._context, "voice_status", status)
        except Exception:
            pass

    def _set_last_text(self, text: str) -> None:
        try:
            setattr(self._context, "voice_last_text", text)
        except Exception:
            pass

    def _set_last_schema(self, schema_line: str) -> None:
        try:
            setattr(self._context, "voice_last_schema", schema_line)
        except Exception:
            pass

    def _set_prompt(self, prompt: str | None) -> None:
        try:
            setattr(self._context, "voice_clarify_prompt", prompt)
        except Exception:
            pass

    def _log(self, event_type: str, **payload) -> None:
        bb = getattr(self._context, "blackbox", None)
        try:
            if bb is not None:
                bb.log_event(event_type, **payload)
        except Exception:
            pass

    # --- audio capture ---
    def _on_audio(self, indata: np.ndarray, frames: int, _t, status) -> None:
        if status:
            self._log("wake_audio_status", status=str(status))
        if self._stop.is_set():
            return
        x = np.asarray(indata).reshape(-1)
        # sounddevice float32 in [-1,1] by default; convert to int16 for wake.
        x_i16 = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
        try:
            self._q.put_nowait(x_i16)
        except queue.Full:
            # Drop oldest by draining one then pushing.
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(x_i16)
            except queue.Full:
                pass

    def _run(self) -> None:
        import sounddevice as sd

        sr = int(self._cfg.sample_rate_hz)
        chunk = int(round(sr * (self._cfg.wake_chunk_ms / 1000.0)))
        if chunk <= 0:
            chunk = 1280

        self._set_status("wake_listening")
        self._log(
            "wake_listening_started",
            sample_rate_hz=sr,
            chunk_samples=chunk,
            wake_confirm_chunks=int(self._cfg.wake_confirm_chunks),
            wake_threshold=float(self._cfg.wake_threshold),
            refractory_ms=int(self._cfg.wake_refractory_ms),
        )

        wake_streak = 0
        refractory_until = 0.0

        # capture state
        capturing = False
        cap_chunks: list[np.ndarray] = []
        cap_start = 0.0
        cap_silence_streak = 0

        min_chunks = max(1, int(np.ceil(self._cfg.capture_min_s / (self._cfg.wake_chunk_ms / 1000.0))))
        max_chunks = max(min_chunks, int(np.ceil(self._cfg.capture_max_s / (self._cfg.wake_chunk_ms / 1000.0))))
        hang_chunks = max(1, int(np.ceil(self._cfg.capture_silence_hangover_ms / self._cfg.wake_chunk_ms)))

        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=chunk,
            callback=self._on_audio,
            device=self._cfg.device,
        ):
            while not self._stop.is_set():
                try:
                    x_i16 = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue

                now = time.time()

                if capturing:
                    cap_chunks.append(x_i16)
                    # VAD on chunk (float32)
                    x_f = x_i16.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(x_f)))) if x_f.size else 0.0
                    is_speech = rms >= float(self._cfg.vad_rms_threshold)
                    cap_silence_streak = 0 if is_speech else (cap_silence_streak + 1)

                    if len(cap_chunks) < min_chunks:
                        continue
                    if cap_silence_streak >= hang_chunks or len(cap_chunks) >= max_chunks:
                        # finalize capture → ASR once
                        capturing = False
                        self._set_status("asr")
                        pcm_i16 = np.concatenate(cap_chunks, axis=0)
                        dur_s = float(pcm_i16.size) / float(sr)
                        self._log(
                            "wake_capture_end",
                            duration_s=dur_s,
                            n_chunks=len(cap_chunks),
                            end_reason="silence" if cap_silence_streak >= hang_chunks else "max_len",
                        )
                        record_voice_obs(self._context, wake_captures=1)

                        try:
                            r = self._asr.transcribe(
                                pcm_i16,
                                lang_hint=self._cfg.lang_hint,
                                sample_rate_hz=sr,
                            )
                        except Exception as e:
                            self._set_status(f"asr_error: {e}")
                            self._log("voice_asr_error", error=str(e))
                            record_voice_obs(self._context, voice_asr_errors=1)
                            refractory_until = time.time() + (self._cfg.wake_refractory_ms / 1000.0)
                            self._set_status("wake_cooldown")
                            cap_chunks = []
                            continue

                        self._set_last_text(r.text or "")
                        self._log(
                            "voice_asr_done",
                            text=r.text,
                            t_asr_ms=int(r.t_asr_ms),
                            language=r.language,
                            duration_audio_s=r.duration_audio_s,
                            n_segments=len(r.segments),
                        )
                        record_voice_obs(self._context, voice_asr_done=1)

                        # Intent + executor (mostly copy from PTT worker)
                        self._set_status("intent")
                        raw_in = (r.text or "").strip()
                        t_in = normalize_zh_command_text(raw_in).strip()
                        if raw_in != t_in and raw_in:
                            self._log("voice_text_normalized", before=raw_in, after=t_in)
                        self._set_last_text(t_in or raw_in)

                        cr = self._clarify.feed(
                            t_in,
                            use_llm=bool(self._cfg.use_llm),
                            llm_client=self._llm_client,
                            llm_textfix_client=self._llm_textfix_client,
                            llm_bundle_client=self._llm_bundle_client,
                            enable_clarify=bool(self._cfg.enable_clarify),
                            asr_low_confidence=False,
                            llm_only_on_asr_low_confidence=bool(self._cfg.llm_only_on_asr_low_confidence),
                        )

                        if cr.phase == "clarify":
                            # In always-on mode, we still show the prompt, but we do not keep capturing
                            # until the user says the wakeword again. (Session 31 keeps wakeword as the gate.)
                            self._set_prompt(cr.clarify_prompt_zh)
                            self._set_status("clarify_wait")
                            self._set_last_schema("")
                            self._log("voice_clarify_prompt", prompt=cr.clarify_prompt_zh, note=cr.note)
                            record_voice_obs(self._context, voice_clarify_prompts=1)
                            refractory_until = time.time() + (self._cfg.wake_refractory_ms / 1000.0)
                            self._set_status("wake_cooldown")
                            cap_chunks = []
                            continue

                        self._set_prompt(None)

                        non_noop = [c for c in (cr.commands or ()) if c.get("cmd") != "noop"]
                        schema_line = "无法解析" if not non_noop else "; ".join(
                            [c.get("cmd", "?") for c in non_noop[:3]]
                        )
                        self._set_last_schema(schema_line)
                        self._log(
                            "voice_intent_resolved",
                            parser=cr.parser,
                            note=cr.note,
                            commands=list(cr.commands),
                        )
                        if not cr.commands:
                            self._set_status("noop")
                            self._log("voice_noop", reason="no_commands")
                            refractory_until = time.time() + (self._cfg.wake_refractory_ms / 1000.0)
                            self._set_status("wake_cooldown")
                            cap_chunks = []
                            continue

                        self._set_status("execute")
                        execute_voice_commands(cr.commands, self._context)
                        self._set_status("wake_cooldown")
                        self._log("voice_execute_done", n_commands=len(cr.commands))

                        # cooldown then resume scanning
                        refractory_until = time.time() + (self._cfg.wake_refractory_ms / 1000.0)
                        cap_chunks = []
                        continue

                    continue

                # Not capturing: wake scanning only (80 ms chunks)
                if now < refractory_until:
                    continue

                w = self._wake.predict_chunk(x_i16)
                record_voice_obs(self._context, wake_chunks=1)
                if w.detected:
                    wake_streak += 1
                else:
                    wake_streak = 0

                if wake_streak >= int(self._cfg.wake_confirm_chunks):
                    wake_streak = 0
                    capturing = True
                    cap_chunks = []
                    cap_start = time.time()
                    cap_silence_streak = 0
                    self._set_prompt(None)
                    self._set_last_text("")
                    self._set_last_schema("")
                    self._set_status("wake_capturing")
                    self._log(
                        "wake_detected",
                        max_score=float(w.max_score),
                        scores=w.scores,
                        capture_min_s=float(self._cfg.capture_min_s),
                        capture_max_s=float(self._cfg.capture_max_s),
                        hangover_ms=int(self._cfg.capture_silence_hangover_ms),
                    )
                    record_voice_obs(self._context, wake_detections=1)
                    # Put the current chunk into capture buffer so we don't lose leading speech.
                    cap_chunks.append(x_i16)
                    continue


class _StatusShim:
    """
    Reuse VoiceLlmLoggingClient from PTT worker: it expects an object with _set_status/_log.
    """

    def __init__(self, w: AlwaysOnVoiceWorker) -> None:
        self._w = w

    def _set_status(self, status: str) -> None:
        self._w._set_status(status)

    def _log(self, event_type: str, **payload) -> None:
        self._w._log(event_type, **payload)

