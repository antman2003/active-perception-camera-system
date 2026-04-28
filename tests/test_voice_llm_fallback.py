"""Session 30 step 3 — LLM fallback only when rules yield no actionable command."""

from src.voice.intent import parse_text_to_commands
from src.voice.llm_client import MockLlmBundleClient, MockLlmIntentClient, MockLlmTextFixClient


def test_rules_win_home_no_llm():
    llm = MockLlmIntentClient({"cmd": "pan_by", "d_pan_deg": 99.0, "source": "voice"})
    r = parse_text_to_commands("回家", use_llm=True, llm_client=llm)
    assert r.parser == "rule"
    assert r.commands == ({"cmd": "home", "source": "voice"},)


def test_llm_on_noop_gibberish():
    llm = MockLlmIntentClient({"cmd": "home", "source": "voice"})
    r = parse_text_to_commands("xyz nonsense", use_llm=True, llm_client=llm)
    assert r.parser == "llm"
    assert r.commands[0]["cmd"] == "home"


def test_llm_invalid_returns_rules_noop():
    llm = MockLlmIntentClient({"cmd": "move_to", "source": "voice"})
    r = parse_text_to_commands("xyz", use_llm=True, llm_client=llm)
    assert r.parser == "none"
    assert r.commands[0]["cmd"] == "noop"
    assert r.note == "llm_failed"


def test_llm_normalizes_common_delta_fields():
    llm = MockLlmIntentClient(
        {"cmd": "move_by", "delta_pan_deg": 10, "delta_tilt_deg": -5, "source": "voice"}
    )
    r = parse_text_to_commands("xyz", use_llm=True, llm_client=llm)
    assert r.parser == "llm"
    assert r.commands == (
        {"cmd": "move_by", "d_pan_deg": 10.0, "d_tilt_deg": -5.0, "source": "voice"},
    )


def test_llm_noop_output_keeps_rules_noop():
    llm = MockLlmIntentClient({"cmd": "noop", "reason": "llm_unsure"})
    r = parse_text_to_commands("xyz", use_llm=True, llm_client=llm)
    assert r.parser == "none"
    assert r.commands[0]["cmd"] == "noop"
    assert r.note == "llm_failed"


def test_llm_only_on_low_confidence_requires_flag():
    llm = MockLlmIntentClient({"cmd": "home", "source": "voice"})
    r = parse_text_to_commands(
        "xyz",
        use_llm=True,
        llm_client=llm,
        asr_low_confidence=False,
        llm_only_on_asr_low_confidence=True,
    )
    assert r.parser == "none"

    r2 = parse_text_to_commands(
        "xyz",
        use_llm=True,
        llm_client=llm,
        asr_low_confidence=True,
        llm_only_on_asr_low_confidence=True,
    )
    assert r2.parser == "llm"


def test_llm_textfix_then_rules_win():
    # Rules fail on the raw ASR text; LLM fixes to a clean command sentence; then rules win.
    llm_cmd = MockLlmIntentClient({"cmd": "home", "source": "voice"})  # should not be used
    llm_bundle = MockLlmBundleClient({"fixed_text": "向右转五度"})
    r = parse_text_to_commands(
        "向右转舞路",
        use_llm=True,
        llm_client=llm_cmd,
        llm_bundle_client=llm_bundle,
    )
    assert r.parser == "rule"
    assert r.note == "llm_bundle_textfix"
    assert r.commands == ({"cmd": "pan_by", "d_pan_deg": 5.0, "source": "voice"},)


def test_llm_bundle_direct_command_wins():
    llm_bundle = MockLlmBundleClient(
        {"fixed_text": "向右转五度", "command": {"cmd": "pan_by", "d_pan_deg": 5}}
    )
    r = parse_text_to_commands(
        "随便一句",
        use_llm=True,
        llm_client=MockLlmIntentClient({"cmd": "home"}),  # not used
        llm_bundle_client=llm_bundle,
    )
    assert r.parser == "llm"
    assert r.note == "llm_bundle"
    assert r.commands == ({"cmd": "pan_by", "d_pan_deg": 5.0, "source": "voice"},)
