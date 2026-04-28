"""
Session 30 — Step 1: frozen Command JSON subset + validation.

Semantics (degrees, relative motion only):
  - pan:  left = negative, right = positive
  - tilt: up = positive, down = negative

When the user is vague ("nudge / rotate a bit"), the *parser* should prefer pan;
this module only validates already-structured JSON.

Sequential phrasing ("先…再…") maps to **multiple single-command objects** in order
(e.g. two `pan_by` deltas), not one merged delta. The executor (or a thin queue layer)
runs them one after another relative to the pose after each step — do **not** sum
angles into one JSON unless the user explicitly asked for a net offset only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

# Single-action cap per axis (Session 30 plan).
MAX_DELTA_DEG: Final[float] = 60.0

VOICE_CMD_HOME: Final[str] = "home"
VOICE_CMD_SEARCH: Final[str] = "search"
VOICE_CMD_NOOP: Final[str] = "noop"
VOICE_CMD_MOVE_BY: Final[str] = "move_by"
VOICE_CMD_PAN_BY: Final[str] = "pan_by"
VOICE_CMD_TILT_BY: Final[str] = "tilt_by"

ALLOWED_CMDS: frozenset[str] = frozenset(
    {
        VOICE_CMD_HOME,
        VOICE_CMD_SEARCH,
        VOICE_CMD_NOOP,
        VOICE_CMD_MOVE_BY,
        VOICE_CMD_PAN_BY,
        VOICE_CMD_TILT_BY,
    }
)

DEFAULT_SOURCE = "voice"
ALLOWED_SOURCES: frozenset[str] = frozenset({DEFAULT_SOURCE, "gesture", "test"})


@dataclass(frozen=True)
class VoiceCommandValidation:
    ok: bool
    error_code: str | None
    message: str | None
    """Human-readable reason when ok is False."""

    normalized: dict[str, Any] | None
    """Canonical dict safe to hand to executor when ok is True."""


def _is_real_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def _as_float(x: Any) -> float:
    return float(x)


def _reject(code: str, msg: str) -> VoiceCommandValidation:
    return VoiceCommandValidation(False, code, msg, None)


def _accept(normalized: dict[str, Any]) -> VoiceCommandValidation:
    return VoiceCommandValidation(True, None, None, normalized)


def validate_voice_command(obj: Any) -> VoiceCommandValidation:
    """
    Validate a Command JSON object. Unknown top-level keys are rejected.
    Does not touch hardware.
    """
    if not isinstance(obj, dict):
        return _reject("E_NOT_OBJECT", "payload must be a JSON object")

    if "cmd" not in obj:
        return _reject("E_CMD_MISSING", "missing required field 'cmd'")

    cmd = obj["cmd"]
    if not isinstance(cmd, str):
        return _reject("E_CMD_TYPE", "'cmd' must be a string")

    if cmd not in ALLOWED_CMDS:
        return _reject("E_CMD_UNKNOWN", f"unknown cmd {cmd!r}")

    src = obj.get("source", DEFAULT_SOURCE)
    if src is not None and not isinstance(src, str):
        return _reject("E_SOURCE_TYPE", "'source' must be a string")
    if isinstance(src, str) and src not in ALLOWED_SOURCES:
        return _reject("E_SOURCE_UNKNOWN", f"unknown source {src!r}")

    allowed_keys: set[str]
    if cmd == VOICE_CMD_HOME:
        allowed_keys = {"cmd", "source"}
    elif cmd == VOICE_CMD_SEARCH:
        allowed_keys = {"cmd", "source"}
    elif cmd == VOICE_CMD_NOOP:
        allowed_keys = {"cmd", "source", "reason"}
    elif cmd == VOICE_CMD_PAN_BY:
        allowed_keys = {"cmd", "source", "d_pan_deg"}
    elif cmd == VOICE_CMD_TILT_BY:
        allowed_keys = {"cmd", "source", "d_tilt_deg"}
    else:  # move_by
        allowed_keys = {"cmd", "source", "d_pan_deg", "d_tilt_deg"}

    extra = set(obj.keys()) - allowed_keys
    if extra:
        return _reject("E_FIELD_UNKNOWN", f"unknown fields: {sorted(extra)!r}")

    if cmd == VOICE_CMD_HOME:
        return _accept({"cmd": VOICE_CMD_HOME, "source": src})

    if cmd == VOICE_CMD_SEARCH:
        return _accept({"cmd": VOICE_CMD_SEARCH, "source": src})

    if cmd == VOICE_CMD_NOOP:
        reason = obj.get("reason")
        if reason is not None and not isinstance(reason, str):
            return _reject("E_REASON_TYPE", "'reason' must be a string")
        out: dict[str, Any] = {"cmd": VOICE_CMD_NOOP, "source": src}
        if isinstance(reason, str) and reason:
            out["reason"] = reason
        return _accept(out)

    if cmd == VOICE_CMD_PAN_BY:
        if "d_pan_deg" not in obj:
            return _reject("E_FIELD_MISSING", "pan_by requires 'd_pan_deg'")
        if not _is_real_number(obj["d_pan_deg"]):
            return _reject("E_FIELD_TYPE", "'d_pan_deg' must be a finite number")
        dp = _as_float(obj["d_pan_deg"])
        if abs(dp) > MAX_DELTA_DEG:
            return _reject(
                "E_FIELD_RANGE",
                f"|d_pan_deg| must be <= {MAX_DELTA_DEG:g} (got {dp})",
            )
        if dp == 0.0:
            return _reject("E_PAN_ZERO", "pan_by with d_pan_deg==0 is invalid; use noop")
        return _accept({"cmd": VOICE_CMD_PAN_BY, "d_pan_deg": dp, "source": src})

    if cmd == VOICE_CMD_TILT_BY:
        if "d_tilt_deg" not in obj:
            return _reject("E_FIELD_MISSING", "tilt_by requires 'd_tilt_deg'")
        if not _is_real_number(obj["d_tilt_deg"]):
            return _reject("E_FIELD_TYPE", "'d_tilt_deg' must be a finite number")
        dt = _as_float(obj["d_tilt_deg"])
        if abs(dt) > MAX_DELTA_DEG:
            return _reject(
                "E_FIELD_RANGE",
                f"|d_tilt_deg| must be <= {MAX_DELTA_DEG:g} (got {dt})",
            )
        if dt == 0.0:
            return _reject(
                "E_TILT_ZERO", "tilt_by with d_tilt_deg==0 is invalid; use noop"
            )
        return _accept({"cmd": VOICE_CMD_TILT_BY, "d_tilt_deg": dt, "source": src})

    # move_by: relative combined; either axis may be omitted -> 0, but not both zero.
    has_pan = "d_pan_deg" in obj
    has_tilt = "d_tilt_deg" in obj
    if not has_pan and not has_tilt:
        return _reject(
            "E_MOVE_NEED_AXIS",
            "move_by requires at least one of 'd_pan_deg', 'd_tilt_deg'",
        )

    dp = 0.0
    dt = 0.0
    if has_pan:
        if not _is_real_number(obj["d_pan_deg"]):
            return _reject("E_FIELD_TYPE", "'d_pan_deg' must be a finite number")
        dp = _as_float(obj["d_pan_deg"])
        if abs(dp) > MAX_DELTA_DEG:
            return _reject(
                "E_FIELD_RANGE",
                f"|d_pan_deg| must be <= {MAX_DELTA_DEG:g} (got {dp})",
            )
    if has_tilt:
        if not _is_real_number(obj["d_tilt_deg"]):
            return _reject("E_FIELD_TYPE", "'d_tilt_deg' must be a finite number")
        dt = _as_float(obj["d_tilt_deg"])
        if abs(dt) > MAX_DELTA_DEG:
            return _reject(
                "E_FIELD_RANGE",
                f"|d_tilt_deg| must be <= {MAX_DELTA_DEG:g} (got {dt})",
            )

    if dp == 0.0 and dt == 0.0:
        return _reject(
            "E_MOVE_BOTH_ZERO",
            "move_by with both deltas zero is invalid; use noop",
        )

    return _accept(
        {
            "cmd": VOICE_CMD_MOVE_BY,
            "d_pan_deg": dp,
            "d_tilt_deg": dt,
            "source": src,
        }
    )


def validate_intent(obj: Any) -> bool:
    """Plan-compatible boolean API."""
    return validate_voice_command(obj).ok
