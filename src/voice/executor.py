"""
Session 30 — Step 4: Execute validated voice commands.

This module is the *only* place that may translate a validated Command JSON into
hardware calls (pan-tilt) or state-machine triggers (search flag).

It must be safe when hardware is missing (``pan_tilt is None``) and must never
raise on normal "no hardware" runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.voice.metrics import record_voice_obs


@runtime_checkable
class _BlackboxLike(Protocol):
    def log_event(self, event_type: str, **payload) -> None: ...


@runtime_checkable
class _PanTiltLike(Protocol):
    def home(self, smooth: bool = True, settle_s: float | None = None) -> str: ...

    def move_by(
        self,
        delta_pan: float = 0.0,
        delta_tilt: float = 0.0,
        smooth: bool = False,
        settle_s: float | None = None,
        step_deg: int | None = None,
        step_settle_s: float | None = None,
    ) -> str: ...


@dataclass
class VoiceExecutionReport:
    ok: bool
    action: str
    reason: str | None = None
    wrote_hardware: bool = False
    set_search_flag: bool = False
    pt_suppress_until: float | None = None


def _bb(context: Any) -> _BlackboxLike | None:
    b = getattr(context, "blackbox", None)
    return b if isinstance(b, _BlackboxLike) else None


def _pt(context: Any) -> _PanTiltLike | None:
    p = getattr(context, "pan_tilt", None)
    return p if isinstance(p, _PanTiltLike) else None


def execute_voice_command(
    cmd: dict[str, Any],
    context: Any,
    *,
    pt_suppress_s: float = 0.6,
    smooth: bool = True,
) -> VoiceExecutionReport:
    """
    Execute one **validated** command dict.

    Expected cmd schema: from ``validate_voice_command(...).normalized``.
    """
    c = (cmd or {}).get("cmd")
    bb = _bb(context)
    pt = _pt(context)

    # --- noop ---
    if c == "noop":
        if bb is not None:
            bb.log_event("voice_command_noop", reason=cmd.get("reason"))
        record_voice_obs(context, voice_exec_noop=1)
        return VoiceExecutionReport(True, "noop", cmd.get("reason"), False, False, None)

    # --- search flag (FSM trigger) ---
    if c == "search":
        try:
            setattr(context, "voice_request_search", True)
        except Exception:
            pass
        if bb is not None:
            bb.log_event("voice_command_search", cmd=cmd)
        record_voice_obs(context, voice_exec_search=1)
        return VoiceExecutionReport(True, "search", None, False, True, None)

    # --- pan-tilt actions (safe when missing hardware) ---
    if pt is None:
        if bb is not None:
            bb.log_event("voice_command_skipped_no_pantilt", cmd=cmd)
        record_voice_obs(context, voice_exec_skipped_no_pantilt=1)
        return VoiceExecutionReport(False, str(c), "no_pan_tilt", False, False, None)

    suppress_until = time.time() + float(pt_suppress_s)
    try:
        setattr(context, "voice_pt_suppress_until", suppress_until)
    except Exception:
        pass

    # Log before act.
    if bb is not None:
        bb.log_event("voice_command_execute", cmd=cmd, pt_suppress_s=float(pt_suppress_s))

    try:
        if c == "home":
            pt.home(smooth=smooth)
            record_voice_obs(context, voice_exec_hardware=1)
            return VoiceExecutionReport(True, "home", None, True, False, suppress_until)

        if c == "pan_by":
            dp = float(cmd.get("d_pan_deg", 0.0))
            pt.move_by(delta_pan=dp, delta_tilt=0.0, smooth=smooth)
            record_voice_obs(context, voice_exec_hardware=1)
            return VoiceExecutionReport(True, "pan_by", None, True, False, suppress_until)

        if c == "tilt_by":
            dt = float(cmd.get("d_tilt_deg", 0.0))
            pt.move_by(delta_pan=0.0, delta_tilt=dt, smooth=smooth)
            record_voice_obs(context, voice_exec_hardware=1)
            return VoiceExecutionReport(True, "tilt_by", None, True, False, suppress_until)

        if c == "move_by":
            dp = float(cmd.get("d_pan_deg", 0.0))
            dt = float(cmd.get("d_tilt_deg", 0.0))
            pt.move_by(delta_pan=dp, delta_tilt=dt, smooth=smooth)
            record_voice_obs(context, voice_exec_hardware=1)
            return VoiceExecutionReport(True, "move_by", None, True, False, suppress_until)

        # Unknown cmd should never happen if schema validation is honored.
        if bb is not None:
            bb.log_event("voice_command_rejected", reason="unexpected_cmd", cmd=cmd)
        record_voice_obs(context, voice_exec_rejected=1)
        return VoiceExecutionReport(False, str(c), "unexpected_cmd", False, False, suppress_until)
    except Exception as e:
        if bb is not None:
            bb.log_event("voice_command_failed", error=str(e), cmd=cmd)
        record_voice_obs(context, voice_exec_failed=1)
        return VoiceExecutionReport(False, str(c), str(e), False, False, suppress_until)


def execute_voice_commands(
    commands: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    context: Any,
    *,
    pt_suppress_s: float = 0.6,
    smooth: bool = True,
) -> list[VoiceExecutionReport]:
    """
    Execute a list/tuple of validated commands sequentially.
    """
    out: list[VoiceExecutionReport] = []
    for cmd in list(commands):
        out.append(
            execute_voice_command(
                cmd, context, pt_suppress_s=pt_suppress_s, smooth=smooth
            )
        )
    return out

