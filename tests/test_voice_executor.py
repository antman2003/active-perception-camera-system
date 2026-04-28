"""Session 30 step 4 — executor safety and semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.voice.metrics import VoiceObsCounters
from src.voice.executor import execute_voice_command, execute_voice_commands


@dataclass
class FakeBlackbox:
    events: list[tuple[str, dict]] = field(default_factory=list)

    def log_event(self, event_type: str, **payload) -> None:
        self.events.append((event_type, dict(payload)))


@dataclass
class FakePanTilt:
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def home(self, smooth: bool = True, settle_s=None) -> str:
        self.calls.append(("home", {"smooth": smooth, "settle_s": settle_s}))
        return "OK"

    def move_by(
        self,
        delta_pan: float = 0.0,
        delta_tilt: float = 0.0,
        smooth: bool = False,
        settle_s=None,
        step_deg=None,
        step_settle_s=None,
    ) -> str:
        self.calls.append(
            (
                "move_by",
                {
                    "delta_pan": float(delta_pan),
                    "delta_tilt": float(delta_tilt),
                    "smooth": bool(smooth),
                },
            )
        )
        return "OK"


@dataclass
class Ctx:
    pan_tilt: FakePanTilt | None = None
    blackbox: FakeBlackbox | None = None
    voice_request_search: bool = False
    voice_pt_suppress_until: float = 0.0
    voice_obs: VoiceObsCounters | None = None


def test_noop_does_not_touch_hardware():
    ctx = Ctx(pan_tilt=FakePanTilt(), blackbox=FakeBlackbox())
    r = execute_voice_command({"cmd": "noop", "reason": "x"}, ctx)
    assert r.ok
    assert r.action == "noop"
    assert ctx.pan_tilt.calls == []


def test_search_sets_flag_no_hardware_needed():
    ctx = Ctx(pan_tilt=None, blackbox=FakeBlackbox(), voice_obs=VoiceObsCounters())
    r = execute_voice_command({"cmd": "search", "source": "voice"}, ctx)
    assert r.ok
    assert r.set_search_flag
    assert ctx.voice_request_search is True
    assert ctx.voice_obs is not None
    assert ctx.voice_obs.snapshot().get("voice_exec_search") == 1


def test_home_calls_pantilt_and_sets_suppress_until():
    ctx = Ctx(pan_tilt=FakePanTilt(), blackbox=FakeBlackbox())
    r = execute_voice_command({"cmd": "home", "source": "voice"}, ctx, pt_suppress_s=0.5)
    assert r.ok
    assert r.wrote_hardware
    assert ctx.pan_tilt.calls[0][0] == "home"
    assert ctx.voice_pt_suppress_until > 0.0
    assert r.pt_suppress_until == ctx.voice_pt_suppress_until


def test_move_by_calls_pantilt():
    ctx = Ctx(pan_tilt=FakePanTilt(), blackbox=FakeBlackbox())
    r = execute_voice_command(
        {"cmd": "move_by", "d_pan_deg": -10.0, "d_tilt_deg": 5.0, "source": "voice"},
        ctx,
    )
    assert r.ok
    assert ctx.pan_tilt.calls[-1] == (
        "move_by",
        {"delta_pan": -10.0, "delta_tilt": 5.0, "smooth": True},
    )


def test_pan_by_and_tilt_by():
    ctx = Ctx(pan_tilt=FakePanTilt(), blackbox=FakeBlackbox())
    execute_voice_command({"cmd": "pan_by", "d_pan_deg": 12.0, "source": "voice"}, ctx)
    execute_voice_command({"cmd": "tilt_by", "d_tilt_deg": -7.0, "source": "voice"}, ctx)
    assert ctx.pan_tilt.calls[-2][1]["delta_pan"] == 12.0
    assert ctx.pan_tilt.calls[-2][1]["delta_tilt"] == 0.0
    assert ctx.pan_tilt.calls[-1][1]["delta_pan"] == 0.0
    assert ctx.pan_tilt.calls[-1][1]["delta_tilt"] == -7.0


def test_missing_pantilt_is_safe():
    ctx = Ctx(pan_tilt=None, blackbox=FakeBlackbox())
    r = execute_voice_command({"cmd": "home", "source": "voice"}, ctx)
    assert not r.ok
    assert r.reason == "no_pan_tilt"


def test_execute_sequence_runs_in_order():
    ctx = Ctx(pan_tilt=FakePanTilt(), blackbox=FakeBlackbox())
    reports = execute_voice_commands(
        [
            {"cmd": "pan_by", "d_pan_deg": -20.0, "source": "voice"},
            {"cmd": "pan_by", "d_pan_deg": 10.0, "source": "voice"},
        ],
        ctx,
    )
    assert [r.action for r in reports] == ["pan_by", "pan_by"]
    assert ctx.pan_tilt.calls[0][1]["delta_pan"] == -20.0
    assert ctx.pan_tilt.calls[1][1]["delta_pan"] == 10.0

