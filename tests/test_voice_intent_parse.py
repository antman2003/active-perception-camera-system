"""Session 30 step 3 — rule-based text → Command JSON (+ schema)."""

import pytest

from src.voice.intent import parse_text_to_command, parse_text_to_commands


def _cmds(text: str) -> tuple[dict, ...]:
    return parse_text_to_commands(text).commands


@pytest.mark.parametrize(
    "text,expected",
    [
        ("回家", ({"cmd": "home", "source": "voice"},)),
        ("go home", ({"cmd": "home", "source": "voice"},)),
        ("复位", ({"cmd": "home", "source": "voice"},)),
        ("找人", ({"cmd": "search", "source": "voice"},)),
        ("帮我扫描一下", ({"cmd": "search", "source": "voice"},)),
        ("search", ({"cmd": "search", "source": "voice"},)),
        ("find people", ({"cmd": "search", "source": "voice"},)),
        ("左转20度", ({"cmd": "pan_by", "d_pan_deg": -20.0, "source": "voice"},)),
        ("向右转十度", ({"cmd": "pan_by", "d_pan_deg": 10.0, "source": "voice"},)),
        ("向上三十度", ({"cmd": "tilt_by", "d_tilt_deg": 30.0, "source": "voice"},)),
        ("向下轉5度", ({"cmd": "tilt_by", "d_tilt_deg": -5.0, "source": "voice"},)),
        ("pan 50 tilt -30", ({"cmd": "move_by", "d_pan_deg": 50.0, "d_tilt_deg": -30.0, "source": "voice"},)),
        ("pan -10", ({"cmd": "move_by", "d_pan_deg": -10.0, "d_tilt_deg": 0.0, "source": "voice"},)),
        ("tilt 15", ({"cmd": "move_by", "d_pan_deg": 0.0, "d_tilt_deg": 15.0, "source": "voice"},)),
        ("转一点", ({"cmd": "pan_by", "d_pan_deg": 5.0, "source": "voice"},)),
        ("向左转一点", ({"cmd": "pan_by", "d_pan_deg": -5.0, "source": "voice"},)),
        ("move_by pan 10 tilt -5", ({"cmd": "move_by", "d_pan_deg": 10.0, "d_tilt_deg": -5.0, "source": "voice"},)),
        ("左转999度", ({"cmd": "pan_by", "d_pan_deg": -60.0, "source": "voice"},)),
        ("回家，然后左转10度", ({"cmd": "home", "source": "voice"}, {"cmd": "pan_by", "d_pan_deg": -10.0, "source": "voice"})),
        ("先扫描再向右轉十五度", ({"cmd": "search", "source": "voice"}, {"cmd": "pan_by", "d_pan_deg": 15.0, "source": "voice"})),
        ("Turn left 15 degrees", ({"cmd": "pan_by", "d_pan_deg": -15.0, "source": "voice"},)),
        ("tilt up 20 deg", ({"cmd": "tilt_by", "d_tilt_deg": 20.0, "source": "voice"},)),
        (
            "胡言乱语 banana",
            ({"cmd": "noop", "source": "voice", "reason": "no matching rule"},),
        ),
        (
            "刚才做得很好，请向右再转三度",
            ({"cmd": "pan_by", "d_pan_deg": 3.0, "source": "voice"},),
        ),
    ],
    ids=[
        "zh_home",
        "en_home",
        "zh_reset",
        "zh_search",
        "zh_scan",
        "en_search_kw",
        "en_find_people",
        "zh_pan_left_arabic",
        "zh_pan_right_cn_deg",
        "zh_tilt_up_cn",
        "zh_tilt_down_mixed",
        "en_move_by_both",
        "en_pan_only",
        "en_tilt_only",
        "vague_pan",
        "vague_pan_left",
        "en_move_by_phrase",
        "clamp_pan",
        "home_then_pan",
        "search_then_pan",
        "en_turn_left_degrees",
        "en_tilt_up",
        "noop_gibberish",
        "polite_right_again",
    ],
)
def test_parse_text_to_commands_table(text, expected):
    assert _cmds(text) == expected


def test_offline_zh_normalization_fixes_asr_confusion():
    # Typical ASR confusion: "转五度" -> "撞武度"
    assert _cmds("向右撞武度") == (
        {"cmd": "pan_by", "d_pan_deg": 5.0, "source": "voice"},
    )
    # Another common confusion in our logs: 五度 -> 无度/有度 (only in rotate+degree context)
    assert _cmds("然后再向右转无度") == (
        {"cmd": "pan_by", "d_pan_deg": 5.0, "source": "voice"},
    )
    # Another common confusion: 五度 -> 舞度
    assert _cmds("然后向上转舞度") == (
        {"cmd": "tilt_by", "d_tilt_deg": 5.0, "source": "voice"},
    )


def test_parse_text_to_commands_recording_style():
    t = "請向左轉 二十度 然後再向右轉 十度完了之後再向上轉 十度向下轉十度"
    assert _cmds(t) == (
        {"cmd": "pan_by", "d_pan_deg": -20.0, "source": "voice"},
        {"cmd": "pan_by", "d_pan_deg": 10.0, "source": "voice"},
        {"cmd": "tilt_by", "d_tilt_deg": 10.0, "source": "voice"},
        {"cmd": "tilt_by", "d_tilt_deg": -10.0, "source": "voice"},
    )


def test_two_motions_one_clause():
    assert _cmds("向右轉十度向下轉十度") == (
        {"cmd": "pan_by", "d_pan_deg": 10.0, "source": "voice"},
        {"cmd": "tilt_by", "d_tilt_deg": -10.0, "source": "voice"},
    )


def test_parse_text_to_command_first_non_noop():
    assert parse_text_to_command("胡言乱语") is None
    assert parse_text_to_command("回家") == {"cmd": "home", "source": "voice"}
    assert parse_text_to_command("找人然后左转5度") == {"cmd": "search", "source": "voice"}
