"""
Session 31 — always-on wake word + capture + ASR pipeline.

Data flow:
- Wake scanning: every 80 ms (1280 @16k) chunk goes into wake model only.
- After wake confirmed: capture a full command window (min duration + silence hangover + max cap),
  then send the *whole* captured audio to ASR once.
- After ASR + intent + executor: return to wake scanning with a short refractory cooldown.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import traceback

from src.voice.audio_io import MicChunkStream, MicStreamConfig, chunk_samples
from src.voice.asr_provider import AsrProvider, FasterWhisperAsrProvider
from src.voice.clarify import VoiceIntentClarifySession
from src.voice.executor import execute_voice_commands
from src.voice.intent import normalize_zh_command_text
from src.voice.metrics import record_voice_obs
from src.voice.llm_client import OllamaIntentClient
from src.voice.worker import VoiceLlmLoggingClient
from src.voice.wakeword import MockWakeWordDetector, OpenWakeWordDetector, WakeWordDetector

_SCHEMA_HOLD_AFTER_EXECUTE_S = 8.0
_SCHEMA_HOLD_UNSET = object()

# Keep ASR text briefly even if we quickly return to listening.
_TEXT_HOLD_AFTER_ASR_S = 4.0

# After leaving cooldown, require a short "quiet" gap before re-arming wake detection.
# This prevents back-to-back re-triggers caused by trailing speech/noise from the prior command.
_POST_COOLDOWN_QUIET_CHUNKS = 8


def utterance_capture_end_meta(
    *, cap_silence_streak: int, hang_chunks: int
) -> tuple[str, bool]:
    """
    end_reason + truncated for blackbox `voice_segment_captured` / `wake_capture_end`.
    Caller only invokes this when finalize is due (`silence hangover` or `max_chunks`).
    `truncated` is True when the branch is max-length (not hangover silence).
    """
    end_reason = "silence" if cap_silence_streak >= hang_chunks else "max_len"
    truncated = end_reason == "max_len"
    return end_reason, bool(truncated)


@dataclass
class AlwaysOnVoiceConfig:
    # Audio IO
    sample_rate_hz: int = 16000
    wake_chunk_ms: int = 80  # must match openWakeWord best-efficiency chunk
    device: str | int | None = None

    # Wake word
    wake_backend: str = "openwakeword"  # openwakeword | mock
    wake_threshold: float = 0.5
    wake_confirm_chunks: int = 3
    wake_refractory_ms: int = 1200
    wake_inference_framework: str = "onnx"
    wake_models: list[str] | None = None
    mock_trigger_every_chunks: int = 50

    # Capture window
    capture_min_s: float = 0.8
    capture_max_s: float = 6.0
    capture_silence_hangover_ms: int = 2000
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
        self._in_refractory = False
        self._was_in_refractory = False

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VoiceAlwaysOn", daemon=True)

        # Wake model
        backend = (cfg.wake_backend or "openwakeword").lower().strip()
        if backend == "mock":
            self._wake = MockWakeWordDetector(
                period_chunks=int(cfg.mock_trigger_every_chunks),
                streak_len=int(cfg.wake_confirm_chunks),
            )
        else:
            self._wake = OpenWakeWordDetector(
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

        # Throttle capture HUD metering updates (avoid locking the main thread every 80ms chunk).
        self._cap_ui_next_mono = 0.0

    def start(self) -> None:
        # Mark immediately so HUD/blackbox can prove the worker was started,
        # even if the background thread fails early.
        self._set_status("wake_starting")
        self._log(
            "wake_worker_start_called",
            wake_backend=str(self._cfg.wake_backend),
            wake_threshold=float(self._cfg.wake_threshold),
            wake_confirm_chunks=int(self._cfg.wake_confirm_chunks),
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)

    # --- HUD / blackbox helpers (match PTT names where possible) ---
    def _set_status(self, status: str) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if callable(fn):
                fn(status=status)
            else:
                setattr(self._context, "voice_status", status)
        except Exception:
            pass

    def _set_last_text(self, text: str) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if callable(fn):
                fn(last_text=text)
            else:
                setattr(self._context, "voice_last_text", text)
        except Exception:
            pass

    def _set_last_text_hold(self, text: str, *, hold_s: float) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if callable(fn):
                fn(last_text=text, text_hold_s=float(hold_s))
            else:
                setattr(self._context, "voice_last_text", text)
        except Exception:
            pass

    def _set_last_schema(self, schema_line: str, *, schema_hold_s: Any = _SCHEMA_HOLD_UNSET) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if callable(fn):
                if schema_hold_s is _SCHEMA_HOLD_UNSET:
                    fn(last_schema=schema_line)
                else:
                    fn(last_schema=schema_line, schema_hold_s=schema_hold_s)
            else:
                setattr(self._context, "voice_last_schema", schema_line)
        except Exception:
            pass

    def _set_prompt(self, prompt: str | None) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if callable(fn):
                fn(clarify_prompt=prompt)
            else:
                setattr(self._context, "voice_clarify_prompt", prompt)
        except Exception:
            pass

    def _publish_capture_meter(self, rms_e: float, speech_started: bool) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if not callable(fn):
                return
            fn(capture_rms_e=float(rms_e), capture_speech_started=bool(speech_started))
        except Exception:
            pass

    def _clear_capture_meter(self) -> None:
        try:
            fn = getattr(self._context, "set_voice_ui", None)
            if not callable(fn):
                return
            fn(capture_rms_e=float("nan"), capture_speech_started=False)
        except Exception:
            pass

    def _clear_voice_result_hud(self, *, clear_prompt: bool = True) -> None:
        """
        Clear transcript/schema lines on HUD between wake cycles.

        We intentionally do NOT clear status here; callers set status explicitly.
        """
        self._set_last_text("")
        self._set_last_schema("")
        if clear_prompt:
            self._set_prompt(None)

    def _log(self, event_type: str, **payload) -> None:
        bb = getattr(self._context, "blackbox", None)
        try:
            if bb is not None:
                bb.log_event(event_type, **payload)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            self._run_impl()
        except Exception as e:  # pragma: no cover
            tb = traceback.format_exc()
            self._set_status(f"wake_worker_crashed: {e}")
            self._log("wake_worker_crashed", error=str(e), traceback=tb)
            # Don't leave stale transcript/schema after a fatal worker crash.
            self._clear_voice_result_hud()

    def _run_impl(self) -> None:
        sr = int(self._cfg.sample_rate_hz)
        chunk = chunk_samples(sr, int(self._cfg.wake_chunk_ms))

        self._clear_voice_result_hud()
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
        self._was_in_refractory = False
        arm_after_quiet = False
        quiet_streak = 0

        # capture state
        capturing = False
        cap_chunks: list[np.ndarray] = []
        cap_start = 0.0
        cap_silence_streak = 0
        speech_started = False

        min_chunks = max(1, int(np.ceil(self._cfg.capture_min_s / (self._cfg.wake_chunk_ms / 1000.0))))
        max_chunks = max(min_chunks, int(np.ceil(self._cfg.capture_max_s / (self._cfg.wake_chunk_ms / 1000.0))))
        hang_chunks = max(1, int(np.ceil(self._cfg.capture_silence_hangover_ms / self._cfg.wake_chunk_ms)))

        mic_cfg = MicStreamConfig(
            sample_rate_hz=sr,
            chunk_ms=int(self._cfg.wake_chunk_ms),
            device=self._cfg.device,
        )

        with MicChunkStream(mic_cfg) as mic:
            while not self._stop.is_set():
                # Yielded chunks are guaranteed int16 mono 16k and exact chunk length.
                got_any = False
                for x_i16 in mic.chunks(timeout_s=0.25):
                    got_any = True
                    if self._stop.is_set():
                        break

                    st = mic.consume_status()
                    if st:
                        self._log("wake_audio_status", status=str(st))

                    now = time.time()

                    if capturing:
                        cap_chunks.append(x_i16)
                        x_f = x_i16.astype(np.float32) / 32768.0
                        rms = float(np.sqrt(np.mean(np.square(x_f)))) if x_f.size else 0.0
                        is_speech = rms >= float(self._cfg.vad_rms_threshold)
                        if is_speech:
                            speech_started = True
                            cap_silence_streak = 0
                        else:
                            # Before we have any speech, don't count "silence hangover".
                            # This avoids ending capture early when the user pauses briefly
                            # after the wake phrase (common in real use).
                            cap_silence_streak = (
                                (cap_silence_streak + 1) if speech_started else 0
                            )

                        mono = time.monotonic()
                        if mono >= float(self._cap_ui_next_mono):
                            self._publish_capture_meter(rms, speech_started)
                            # ~8–10 Hz is enough for a level meter without spamming HUD updates.
                            self._cap_ui_next_mono = mono + 0.12

                        if len(cap_chunks) < min_chunks:
                            continue

                        # If the user never speaks after wake, keep capturing until max_len.
                        if not speech_started and len(cap_chunks) < max_chunks:
                            continue

                        if not (cap_silence_streak >= hang_chunks or len(cap_chunks) >= max_chunks):
                            continue

                        # finalize capture → ASR once
                        capturing = False
                        self._clear_capture_meter()
                        self._cap_ui_next_mono = 0.0
                        self._set_status("asr")
                        # Avoid leaving stale transcript/schema on-screen while ASR runs.
                        self._clear_voice_result_hud()
                        pcm_i16 = np.concatenate(cap_chunks, axis=0)
                        dur_s = float(pcm_i16.size) / float(sr)
                        end_reason, truncated = utterance_capture_end_meta(
                            cap_silence_streak=cap_silence_streak,
                            hang_chunks=hang_chunks,
                        )
                        duration_ms = int(round(dur_s * 1000.0))
                        self._log(
                            "wake_capture_end",
                            duration_s=dur_s,
                            n_chunks=len(cap_chunks),
                            end_reason=end_reason,
                        )
                        self._log(
                            "voice_segment_captured",
                            samples=int(pcm_i16.size),
                            duration_ms=duration_ms,
                            truncated=bool(truncated),
                            end_reason=end_reason,
                            n_chunks=len(cap_chunks),
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
                            refractory_until = time.time() + (
                                self._cfg.wake_refractory_ms / 1000.0
                            )
                            self._in_refractory = True
                            self._set_status("wake_cooldown")
                            self._clear_voice_result_hud()
                            cap_chunks = []
                            continue

                        # Hold transcript briefly so users can read it even if we quickly return to listening.
                        self._set_last_text_hold(r.text or "", hold_s=_TEXT_HOLD_AFTER_ASR_S)
                        self._log(
                            "voice_asr_done",
                            text=r.text,
                            t_asr_ms=int(r.t_asr_ms),
                            language=r.language,
                            duration_audio_s=r.duration_audio_s,
                            n_segments=len(r.segments),
                        )
                        record_voice_obs(self._context, voice_asr_done=1)

                        # If ASR yields empty text, show a clear HUD message instead of
                        # silently proceeding into parsing/LLM with noop.
                        if not (r.text or "").strip():
                            self._log(
                                "voice_asr_empty",
                                note="empty transcript (likely silence / low volume / noise)",
                                duration_audio_s=r.duration_audio_s,
                            )
                            record_voice_obs(self._context, voice_asr_empty=1)
                            self._set_last_text("（未识别到语音，请重试/说大声一点）")
                            self._set_last_schema("")
                            self._set_prompt(None)
                            refractory_until = time.time() + (
                                self._cfg.wake_refractory_ms / 1000.0
                            )
                            self._in_refractory = True
                            self._set_status("no_speech")
                            cap_chunks = []
                            speech_started = False
                            continue

                        self._set_status("intent")
                        raw_in = (r.text or "").strip()
                        t_in = normalize_zh_command_text(raw_in).strip()
                        if raw_in != t_in and raw_in:
                            self._log("voice_text_normalized", before=raw_in, after=t_in)
                        self._set_last_text_hold(t_in or raw_in, hold_s=_TEXT_HOLD_AFTER_ASR_S)

                        cr = self._clarify.feed(
                            t_in,
                            use_llm=bool(self._cfg.use_llm),
                            llm_client=self._llm_client,
                            llm_textfix_client=self._llm_textfix_client,
                            llm_bundle_client=self._llm_bundle_client,
                            enable_clarify=bool(self._cfg.enable_clarify),
                            asr_low_confidence=False,
                            llm_only_on_asr_low_confidence=bool(
                                self._cfg.llm_only_on_asr_low_confidence
                            ),
                        )

                        if cr.phase == "clarify":
                            # Clear prior transcript/schema, but keep the clarify prompt visible.
                            self._clear_voice_result_hud(clear_prompt=False)
                            self._set_prompt(cr.clarify_prompt_zh)
                            self._set_status("clarify_wait")
                            self._log(
                                "voice_clarify_prompt",
                                prompt=cr.clarify_prompt_zh,
                                note=cr.note,
                            )
                            record_voice_obs(self._context, voice_clarify_prompts=1)
                            refractory_until = time.time() + (
                                self._cfg.wake_refractory_ms / 1000.0
                            )
                            self._in_refractory = True
                            self._set_status("wake_cooldown")
                            cap_chunks = []
                            continue

                        self._set_prompt(None)

                        def _fmt_one(cmd: dict[str, Any]) -> str:
                            c = cmd.get("cmd")
                            if c == "home":
                                return "home"
                            if c == "search":
                                return "search"
                            if c == "pan_by":
                                return f"pan_by({cmd.get('d_pan_deg'):+g}°)"
                            if c == "tilt_by":
                                return f"tilt_by({cmd.get('d_tilt_deg'):+g}°)"
                            if c == "move_by":
                                return f"move_by(pan={cmd.get('d_pan_deg'):+g}°, tilt={cmd.get('d_tilt_deg'):+g}°)"
                            return str(cmd)

                        non_noop = [c for c in (cr.commands or ()) if c.get("cmd") != "noop"]
                        if non_noop:
                            parts = [_fmt_one(c) for c in non_noop[:3]]
                            schema_line = "; ".join(parts)
                            if len(non_noop) > 3:
                                schema_line += f" …(+{len(non_noop) - 3})"
                        else:
                            schema_line = "无法解析"
                        # Keep SCHEMA visible briefly; main thread will auto-revert to `SCHEMA: ------`.
                        self._set_last_schema(schema_line, schema_hold_s=_SCHEMA_HOLD_AFTER_EXECUTE_S)
                        self._log(
                            "voice_intent_resolved",
                            parser=cr.parser,
                            note=cr.note,
                            commands=list(cr.commands),
                        )
                        if not cr.commands:
                            self._set_status("noop")
                            self._log("voice_noop", reason="no_commands")
                            refractory_until = time.time() + (
                                self._cfg.wake_refractory_ms / 1000.0
                            )
                            self._in_refractory = True
                            self._set_status("wake_cooldown")
                            self._clear_voice_result_hud()
                            cap_chunks = []
                            continue

                        self._set_status("execute")
                        execute_voice_commands(cr.commands, self._context)
                        self._log("voice_execute_done", n_commands=len(cr.commands))

                        refractory_until = time.time() + (self._cfg.wake_refractory_ms / 1000.0)
                        self._in_refractory = True
                        self._set_status("wake_cooldown")
                        self._clear_capture_meter()
                        cap_chunks = []
                        speech_started = False
                        continue

                    # Not capturing: wake scanning only (80 ms chunks)
                    if now < refractory_until:
                        if not self._in_refractory:
                            self._in_refractory = True
                            self._set_status("wake_cooldown")
                        self._was_in_refractory = True
                        continue
                    if self._was_in_refractory:
                        # Leaving cooldown: start a fresh listening cycle for HUD observers.
                        self._was_in_refractory = False
                        # Don't forcibly clear transcript/schema here; let the timed SCHEMA expiry
                        # (main thread) decide when to revert to the stable anchor.
                        self._clear_capture_meter()
                        # Require a short quiet gap before we accept a new wake detection.
                        arm_after_quiet = True
                        quiet_streak = 0
                    self._in_refractory = False
                    self._set_status("wake_listening")

                    w = self._wake.predict_chunk(x_i16)
                    record_voice_obs(self._context, wake_chunks=1)
                    if arm_after_quiet:
                        if w.detected:
                            quiet_streak = 0
                            continue
                        quiet_streak += 1
                        if quiet_streak < int(_POST_COOLDOWN_QUIET_CHUNKS):
                            continue
                        arm_after_quiet = False
                        quiet_streak = 0
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
                        speech_started = False
                        self._cap_ui_next_mono = 0.0
                        self._clear_voice_result_hud()
                        self._set_status("wake_capturing")
                        wake_payload = dict(
                            max_score=float(w.max_score),
                            scores=w.scores,
                            capture_min_s=float(self._cfg.capture_min_s),
                            capture_max_s=float(self._cfg.capture_max_s),
                            hangover_ms=int(self._cfg.capture_silence_hangover_ms),
                        )
                        self._log("wake_detected", **wake_payload)
                        # Session 31 / docs: stable `voice_*` name for timelines + external tooling.
                        self._log("voice_wake_detected", **wake_payload)
                        record_voice_obs(self._context, wake_detections=1)
                        cap_chunks.append(x_i16)
                        continue

                if not got_any:
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

