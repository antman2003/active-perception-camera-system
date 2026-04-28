"""One-round clarification session for voice intent."""

from src.voice.clarify import VoiceIntentClarifySession
from src.voice.intent import parse_after_clarification
from src.voice.llm_client import MockLlmIntentClient


def test_first_turn_home_no_clarify():
    s = VoiceIntentClarifySession()
    r = s.feed("回家", enable_clarify=True)
    assert r.phase == "resolved"
    assert r.commands[0]["cmd"] == "home"
    assert not s.awaiting_clarification


def test_first_turn_gibberish_then_clarify_prompt():
    s = VoiceIntentClarifySession()
    r = s.feed("xyz nonsense", enable_clarify=True)
    assert r.phase == "clarify"
    assert r.commands == ()
    assert r.clarify_prompt_zh
    assert s.awaiting_clarification


def test_second_turn_resolves_with_rules():
    s = VoiceIntentClarifySession()
    s.feed("xyz", enable_clarify=True)
    r2 = s.feed("左转十度", enable_clarify=True)
    assert r2.phase == "resolved"
    assert r2.commands[0]["cmd"] == "pan_by"
    assert r2.commands[0]["d_pan_deg"] == -10.0
    assert not s.awaiting_clarification


def test_second_turn_uses_llm_with_mock():
    s = VoiceIntentClarifySession()
    llm = MockLlmIntentClient({"cmd": "home", "source": "voice"})
    s.feed("foobar", enable_clarify=True)
    r2 = s.feed("还是不太清楚", enable_clarify=True, use_llm=True, llm_client=llm)
    assert r2.parser == "llm"
    assert r2.commands[0]["cmd"] == "home"


def test_clarify_disabled_falls_through_to_single_shot_llm():
    s = VoiceIntentClarifySession()
    llm = MockLlmIntentClient({"cmd": "search", "source": "voice"})
    r = s.feed("xyz", enable_clarify=False, use_llm=True, llm_client=llm)
    assert r.phase == "resolved"
    assert r.parser == "llm"
    assert r.commands[0]["cmd"] == "search"


def test_parse_after_clarification_rules_on_followup_first():
    r = parse_after_clarification("foo", "向右轉五度", use_llm=False)
    assert r.parser == "rule"
    assert r.commands[0]["cmd"] == "pan_by"
    assert r.commands[0]["d_pan_deg"] == 5.0


def test_reset_clears_pending():
    s = VoiceIntentClarifySession()
    s.feed("xyz", enable_clarify=True)
    s.reset()
    assert not s.awaiting_clarification
    r = s.feed("回家", enable_clarify=True)
    assert r.commands[0]["cmd"] == "home"
