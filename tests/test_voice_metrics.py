"""Session 30 step 6 — voice observability counters."""

from __future__ import annotations

from dataclasses import dataclass

from src.voice.executor import execute_voice_command
from src.voice.metrics import VoiceObsCounters, record_voice_obs


@dataclass
class Ctx:
    voice_obs: VoiceObsCounters


def test_voice_obs_counters_threadsafe_ordering_in_snapshot():
    obs = VoiceObsCounters()
    obs.add("z", 1)
    obs.add("a", 2)
    snap = obs.snapshot()
    assert snap == {"a": 2, "z": 1}


def test_record_voice_obs_no_context_field():
    class Empty:
        pass

    record_voice_obs(Empty(), voice_x=1)  # no crash


def test_executor_increments_obs_on_noop():
    ctx = Ctx(voice_obs=VoiceObsCounters())
    execute_voice_command({"cmd": "noop", "reason": "t"}, ctx)
    assert ctx.voice_obs.snapshot().get("voice_exec_noop") == 1
