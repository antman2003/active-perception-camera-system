"""
Session 30 — Step 5 helper: background voice worker for PTT.

Design goals:
- No serial writes in this thread; it only calls the *executor* with validated JSON.
- Safe when no microphone (raises in the worker; reported to context status).
- PTT trigger is driven by the main UI thread (OpenCV key press).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.voice.asr_provider import AsrProvider, FasterWhisperAsrProvider
from src.voice.clarify import VoiceIntentClarifySession
from src.voice.executor import execute_voice_commands
from src.voice.llm_client import (
    LlmBundleClient,
    LlmIntentClient,
    LlmTextFixClient,
    OllamaIntentClient,
)
from src.voice.intent import _intent_from_rules, _intent_has_non_noop, normalize_zh_command_text
from src.voice.metrics import record_voice_obs

_SCHEMA_HOLD_UNSET = object()
_SCHEMA_HOLD_AFTER_EXECUTE_S = 8.0


@dataclass
class VoiceWorkerConfig:
    lang_hint: str | None = "zh"
    record_seconds: float = 5.0
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    use_llm: bool = False
    llm_model: str = "qwen2.5:1.5b"
    enable_clarify: bool = True
    llm_only_on_asr_low_confidence: bool = False
    save_wav: bool = False


class VoiceLlmLoggingClient:
    """
    Wrap a LLM client to emit explicit blackbox events + HUD status during calls.
    """

    def __init__(self, base: LlmIntentClient, context: Any, worker: "VoicePttWorker"):
        self._base = base
        self._context = context
        self._worker = worker

    def propose_command(self, user_text: str) -> dict[str, Any] | None:
        # Show on HUD that we are in LLM.
        self._worker._set_status("llm_loading")
        t0 = time.perf_counter()
        self._worker._log(
            "voice_llm_attempted",
            prompt_preview=(user_text or "")[:160],
        )
        raw = None
        err = None
        try:
            raw = self._base.propose_command(user_text)
            return raw
        except Exception as e:
            err = str(e)
            return None
        finally:
            dt_ms = int(round((time.perf_counter() - t0) * 1000))
            record_voice_obs(
                self._context,
                voice_llm_propose_attempts=1,
                voice_llm_propose_ok=1 if raw is not None else 0,
                voice_llm_propose_fail=0 if raw is not None else 1,
            )
            self._worker._log(
                "voice_llm_result",
                ok=raw is not None,
                t_llm_ms=dt_ms,
                error=err,
                raw=raw,
            )
            # Back to intent; caller will set the final status.
            self._worker._set_status("intent")

    def propose_fixed_text(self, user_text: str) -> str | None:
        # Show on HUD that we are in LLM text-fix mode.
        self._worker._set_status("llm_textfix_loading")
        t0 = time.perf_counter()
        self._worker._log(
            "voice_llm_textfix_attempted",
            prompt_preview=(user_text or "")[:160],
        )
        fixed = None
        err = None
        try:
            base = self._base
            if not hasattr(base, "propose_fixed_text"):
                return None
            fixed = base.propose_fixed_text(user_text)  # type: ignore[attr-defined]
            return fixed
        except Exception as e:
            err = str(e)
            return None
        finally:
            dt_ms = int(round((time.perf_counter() - t0) * 1000))
            record_voice_obs(
                self._context,
                voice_llm_textfix_attempts=1,
                voice_llm_textfix_ok=1 if fixed is not None else 0,
                voice_llm_textfix_fail=0 if fixed is not None else 1,
            )
            self._worker._log(
                "voice_llm_textfix_result",
                ok=fixed is not None,
                t_llm_ms=dt_ms,
                error=err,
                fixed_text=fixed,
            )
            self._worker._set_status("intent")

    def propose_bundle(self, user_text: str) -> dict[str, Any] | None:
        self._worker._set_status("llm_bundle_loading")
        t0 = time.perf_counter()
        self._worker._log(
            "voice_llm_bundle_attempted",
            prompt_preview=(user_text or "")[:160],
        )
        raw = None
        err = None
        try:
            base = self._base
            if not hasattr(base, "propose_bundle"):
                return None
            raw = base.propose_bundle(user_text)  # type: ignore[attr-defined]
            return raw
        except Exception as e:
            err = str(e)
            return None
        finally:
            dt_ms = int(round((time.perf_counter() - t0) * 1000))
            record_voice_obs(
                self._context,
                voice_llm_bundle_attempts=1,
                voice_llm_bundle_ok=1 if raw is not None else 0,
                voice_llm_bundle_fail=0 if raw is not None else 1,
            )
            self._worker._log(
                "voice_llm_bundle_result",
                ok=raw is not None,
                t_llm_ms=dt_ms,
                error=err,
                raw=raw,
            )
            self._worker._set_status("intent")


class VoicePttWorker:
    def __init__(self, context: Any, cfg: VoiceWorkerConfig):
        self._context = context
        self._cfg = cfg
        self._stop = threading.Event()
        self._trigger = threading.Event()
        self._thread = threading.Thread(target=self._run, name="VoicePTT", daemon=True)

        self._asr: AsrProvider = FasterWhisperAsrProvider(
            model_size=cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            vad_filter=True,
        )
        self._clarify = VoiceIntentClarifySession()
        self._llm_client: LlmIntentClient | None = None
        self._llm_textfix_client: LlmTextFixClient | None = None
        self._llm_bundle_client: LlmBundleClient | None = None
        if cfg.use_llm:
            base = OllamaIntentClient(model=cfg.llm_model)
            # Wrap for explicit events + HUD state.
            wrapped = VoiceLlmLoggingClient(base, context, self)
            self._llm_client = wrapped
            self._llm_textfix_client = wrapped
            self._llm_bundle_client = wrapped

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._trigger.set()
        self._thread.join(timeout=timeout_s)

    def trigger_once(self) -> None:
        """Called by main thread when user presses the PTT key."""
        self._trigger.set()

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

    def _log(self, event_type: str, **payload) -> None:
        bb = getattr(self._context, "blackbox", None)
        try:
            if bb is not None:
                bb.log_event(event_type, **payload)
        except Exception:
            pass

    def _log_ptt_cycle_end(self, outcome: str) -> None:
        obs = getattr(self._context, "voice_obs", None)
        metrics: dict[str, int] = {}
        try:
            if obs is not None and hasattr(obs, "snapshot"):
                metrics = obs.snapshot()  # type: ignore[assignment]
        except Exception:
            metrics = {}
        self._log("voice_ptt_cycle_end", outcome=outcome, metrics=metrics)

    def _maybe_save_wav(self, pcm: np.ndarray, sr: int, *, tag: str) -> str | None:
        if not self._cfg.save_wav:
            return None
        bb = getattr(self._context, "blackbox", None)
        session_dir = getattr(bb, "session_dir", None) if bb is not None else None
        if session_dir is None:
            return None
        try:
            from pathlib import Path

            import soundfile as sf

            audio_dir = Path(session_dir) / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            path = audio_dir / f"voice_{tag}_{ts}.wav"
            sf.write(str(path), pcm.astype(np.float32), sr)
            return str(path)
        except Exception:
            return None

    def _record(self, seconds: float, samplerate: int = 16000) -> np.ndarray:
        import sounddevice as sd

        frames = max(1, int(float(seconds) * samplerate))
        audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        return audio.reshape(-1)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._trigger.wait(0.25)
            if self._stop.is_set():
                break
            if not self._trigger.is_set():
                continue
            self._trigger.clear()

            cycle_outcome = "incomplete"
            try:
                ptt_followup = bool(getattr(self._clarify, "awaiting_clarification", False))
                self._log(
                    "voice_ptt_triggered",
                    frame_idx=getattr(self._context, "frame_count", None),
                    awaiting_clarification=ptt_followup,
                    record_seconds=float(self._cfg.record_seconds),
                    model_size=self._cfg.model_size,
                    use_llm=bool(self._cfg.use_llm),
                    enable_clarify=bool(self._cfg.enable_clarify),
                )
                record_voice_obs(self._context, voice_ptt_presses=1)
                if ptt_followup:
                    record_voice_obs(self._context, voice_ptt_followup_presses=1)

                # Clear previous subtitle lines immediately when recording starts.
                self._set_prompt(None)
                self._set_last_text("")
                self._set_last_schema("")
                self._set_status("recording")
                try:
                    pcm = self._record(self._cfg.record_seconds)
                except Exception as e:
                    self._set_status(f"mic_error: {e}")
                    self._log("voice_mic_error", error=str(e))
                    record_voice_obs(self._context, voice_mic_errors=1)
                    cycle_outcome = "mic_error"
                    continue

                rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32))))) if pcm.size else 0.0
                peak = float(np.max(np.abs(pcm.astype(np.float32)))) if pcm.size else 0.0
                wav_path = self._maybe_save_wav(pcm, 16000, tag="ptt")
                self._log(
                    "voice_audio_recorded",
                    samples=int(pcm.size),
                    sample_rate_hz=16000,
                    duration_s=float(pcm.size) / 16000.0,
                    rms=rms,
                    peak=peak,
                    wav_path=wav_path,
                )
                record_voice_obs(self._context, voice_audio_chunks=1)

                self._set_status("asr")
                try:
                    r = self._asr.transcribe(
                        pcm,
                        lang_hint=self._cfg.lang_hint,
                        sample_rate_hz=16000,
                    )
                except Exception as e:
                    self._set_status(f"asr_error: {e}")
                    self._log("voice_asr_error", error=str(e))
                    record_voice_obs(self._context, voice_asr_errors=1)
                    cycle_outcome = "asr_error"
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

                # Step 5: one-round clarification session (prompt-only first turn if needed).
                self._set_status("intent")
                # For observability: explicitly log whether LLM will be attempted.
                raw_in = (r.text or "").strip()
                t_in = normalize_zh_command_text(raw_in).strip()
                if raw_in != t_in and raw_in:
                    self._log(
                        "voice_text_normalized",
                        before=raw_in,
                        after=t_in,
                    )
                # Show the normalized text on HUD so users see what will be parsed/executed.
                self._set_last_text(t_in or raw_in)
                llm_will_attempt = False
                if self._cfg.use_llm and self._llm_client is not None:
                    if self._clarify.awaiting_clarification:
                        # Follow-up turn: rules on follow-up, then maybe LLM.
                        # We don't know the follow-up yet, so we only mark that LLM is enabled.
                        llm_will_attempt = False
                    else:
                        rules = _intent_from_rules(t_in)
                        llm_will_attempt = not _intent_has_non_noop(rules.commands)
                self._log(
                    "voice_llm_plan",
                    enabled=bool(self._cfg.use_llm),
                    will_attempt=bool(llm_will_attempt),
                )

                cr = self._clarify.feed(
                    t_in,
                    use_llm=self._cfg.use_llm,
                    llm_client=self._llm_client,
                    llm_textfix_client=self._llm_textfix_client,
                    llm_bundle_client=self._llm_bundle_client,
                    enable_clarify=self._cfg.enable_clarify,
                    asr_low_confidence=False,
                    llm_only_on_asr_low_confidence=self._cfg.llm_only_on_asr_low_confidence,
                )

                if cr.phase == "clarify":
                    self._set_prompt(cr.clarify_prompt_zh)
                    self._set_status("clarify_wait")
                    self._set_last_schema("")
                    self._log(
                        "voice_clarify_prompt",
                        prompt=cr.clarify_prompt_zh,
                        note=cr.note,
                    )
                    record_voice_obs(self._context, voice_clarify_prompts=1)
                    cycle_outcome = "clarify_prompt"
                    continue

                self._set_prompt(None)
                # Update HUD schema line (show all non-noop commands; or "无法解析").
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
                if cr.parser == "rule":
                    record_voice_obs(self._context, voice_intent_parser_rule=1)
                elif cr.parser == "llm":
                    record_voice_obs(self._context, voice_intent_parser_llm=1)
                else:
                    record_voice_obs(self._context, voice_intent_parser_none=1)
                if cr.note == "llm_failed":
                    record_voice_obs(self._context, voice_intent_llm_schema_reject=1)
                    self._log(
                        "voice_intent_schema_reject",
                        reason="llm_output_invalid_or_noop_after_llm",
                        parser=cr.parser,
                    )
                if ptt_followup:
                    record_voice_obs(self._context, voice_clarify_followups_completed=1)

                if not cr.commands:
                    self._set_status("noop")
                    self._log("voice_noop", reason="no_commands")
                    cycle_outcome = "no_commands"
                    continue

                # Execute sequentially (validated JSON only).
                self._set_status("execute")
                execute_voice_commands(cr.commands, self._context)
                self._set_status("idle")
                self._log("voice_execute_done", n_commands=len(cr.commands))
                record_voice_obs(
                    self._context,
                    voice_execute_batches=1,
                    voice_execute_commands=len(cr.commands),
                )
                cycle_outcome = "executed"
            finally:
                self._log_ptt_cycle_end(cycle_outcome)

