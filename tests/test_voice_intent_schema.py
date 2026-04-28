"""Session 30 step 1 — intent schema validation."""

import pytest

from src.voice_intent.schema import (
    MAX_DELTA_DEG,
    validate_intent,
    validate_voice_command,
)


def test_validate_intent_accepts_minimal_examples():
    assert validate_intent({"cmd": "home"}) is True
    assert validate_intent({"cmd": "search", "source": "voice"}) is True
    assert validate_intent({"cmd": "noop", "reason": "unrecognized phrase"}) is True
    assert validate_intent({"cmd": "move_by", "d_pan_deg": -10}) is True
    assert validate_intent({"cmd": "move_by", "d_tilt_deg": 5}) is True
    assert validate_intent({"cmd": "pan_by", "d_pan_deg": 20}) is True
    assert validate_intent({"cmd": "tilt_by", "d_tilt_deg": -15}) is True


def test_move_by_combined():
    r = validate_voice_command(
        {"cmd": "move_by", "d_pan_deg": 10, "d_tilt_deg": -5, "source": "test"}
    )
    assert r.ok
    assert r.normalized == {
        "cmd": "move_by",
        "d_pan_deg": 10.0,
        "d_tilt_deg": -5.0,
        "source": "test",
    }


def test_rejects_unknown_cmd():
    r = validate_voice_command({"cmd": "move_to"})
    assert not r.ok
    assert r.error_code == "E_CMD_UNKNOWN"


def test_rejects_unknown_field():
    r = validate_voice_command({"cmd": "home", "extra": 1})
    assert not r.ok
    assert r.error_code == "E_FIELD_UNKNOWN"


def test_rejects_out_of_range_pan_by():
    r = validate_voice_command({"cmd": "pan_by", "d_pan_deg": MAX_DELTA_DEG + 1})
    assert not r.ok
    assert r.error_code == "E_FIELD_RANGE"


def test_rejects_move_by_both_zero():
    r = validate_voice_command({"cmd": "move_by", "d_pan_deg": 0, "d_tilt_deg": 0})
    assert not r.ok
    assert r.error_code == "E_MOVE_BOTH_ZERO"


def test_rejects_move_by_no_axis():
    r = validate_voice_command({"cmd": "move_by"})
    assert not r.ok
    assert r.error_code == "E_MOVE_NEED_AXIS"


def test_rejects_pan_by_zero_use_noop():
    r = validate_voice_command({"cmd": "pan_by", "d_pan_deg": 0})
    assert not r.ok
    assert r.error_code == "E_PAN_ZERO"


def test_rejects_non_object():
    r = validate_voice_command([])
    assert not r.ok
    assert r.error_code == "E_NOT_OBJECT"


def test_rejects_bool_as_number():
    r = validate_voice_command({"cmd": "pan_by", "d_pan_deg": True})
    assert not r.ok
    assert r.error_code == "E_FIELD_TYPE"
