"""Session 30 step 5 — voice search trigger is read by MonitorState."""

from __future__ import annotations

from src.states import MonitorState


class Ctx:
    # Minimal attributes used by the early-return path.
    def __init__(self) -> None:
        self.voice_request_search = True
    frame_count = 123


def test_monitor_state_transitions_on_voice_request_search():
    s = MonitorState()
    ctx = Ctx()
    nxt = s.update(ctx, None, False, None, None, 0.0, 0.0, {}, 0.0, 0.0)
    assert getattr(nxt, "name", "") == "PHYSICAL_SEARCH"
    assert ctx.voice_request_search is False

