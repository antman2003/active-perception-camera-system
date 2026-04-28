"""Voice worker: LLM wrapper emits explicit events."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.voice.llm_client import MockLlmIntentClient
from src.voice.worker import VoiceLlmLoggingClient, VoicePttWorker, VoiceWorkerConfig


@dataclass
class FakeBlackbox:
    events: list[tuple[str, dict]] = field(default_factory=list)

    def log_event(self, event_type: str, **payload) -> None:
        self.events.append((event_type, dict(payload)))


@dataclass
class Ctx:
    blackbox: FakeBlackbox = field(default_factory=FakeBlackbox)
    voice_status: str = "idle"


def test_llm_wrapper_logs_attempt_and_result():
    ctx = Ctx()
    # Build a minimal worker just to use its logging helpers.
    w = VoicePttWorker(ctx, VoiceWorkerConfig(use_llm=False))
    base = MockLlmIntentClient({"cmd": "home", "source": "voice"})
    wrapped = VoiceLlmLoggingClient(base, ctx, w)
    out = wrapped.propose_command("hello")
    assert out is not None
    types = [t for t, _p in ctx.blackbox.events]
    assert "voice_llm_attempted" in types
    assert "voice_llm_result" in types

