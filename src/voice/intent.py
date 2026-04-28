"""
Session 30 — Step 3: text → validated Command JSON (rules first; LLM hook reserved).

All successful paths run through ``validate_voice_command`` (schema / limits).
Sequential wording yields **multiple** commands in order.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from src.voice.llm_client import (
    LlmBundleClient,
    LlmIntentClient,
    LlmTextFixClient,
    normalize_llm_command,
 )
from src.voice_intent.schema import MAX_DELTA_DEG, validate_voice_command

ParserKind = Literal["rule", "llm", "none"]


@dataclass(frozen=True)
class IntentParseResult:
    commands: tuple[dict[str, Any], ...]
    parser: ParserKind
    note: str | None = None


_CN_DIGIT: Final[dict[str, int]] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def normalize_zh_command_text(text: str) -> str:
    """
    Offline, conservative normalization for common Chinese ASR confusions in our
    command domain (pan/tilt + degrees).

    Goal: improve rule hit-rate without using LLM. This is NOT a general-purpose
    Chinese correction system.
    """
    t = (text or "").strip()
    if not t:
        return t

    # Normalize spaces lightly.
    t = re.sub(r"\s+", " ", t)

    # Apply configurable table (if present).
    rules = _load_zh_normalization_rules()
    for rep in rules["replacements"]:
        t = t.replace(rep["from"], rep["to"])
    for rr in rules["regex_rules"]:
        t = re.sub(rr["pattern"], rr["replacement"], t)

    return t


@lru_cache(maxsize=1)
def _load_zh_normalization_rules() -> dict[str, Any]:
    """
    Load `src/voice/zh_normalization_rules.json` if present; fallback to safe defaults.
    Cached to avoid disk IO on each command.
    """
    defaults: dict[str, Any] = {
        "replacements": [{"from": "武度", "to": "五度"}, {"from": "舞度", "to": "五度"}],
        "regex_rules": [
            {
                "pattern": r"((?:向)?(?:左|右|上|下).{0,3})(撞|装|赚)(?=.{0,6}度)",
                "replacement": r"\1转",
            }
            ,
            {
                "pattern": r"((?:转|轉))\s*(?:有|无)度",
                "replacement": r"\1五度",
            }
        ],
    }
    path = Path(__file__).with_name("zh_normalization_rules.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        reps = raw.get("replacements")
        regs = raw.get("regex_rules")
        if not isinstance(reps, list) or not isinstance(regs, list):
            return defaults
        out_reps: list[dict[str, str]] = []
        for r in reps:
            if not isinstance(r, dict):
                continue
            f = r.get("from")
            to = r.get("to")
            if isinstance(f, str) and isinstance(to, str) and f:
                out_reps.append({"from": f, "to": to})
        out_regs: list[dict[str, str]] = []
        for rr in regs:
            if not isinstance(rr, dict):
                continue
            p = rr.get("pattern")
            repl = rr.get("replacement")
            if isinstance(p, str) and isinstance(repl, str) and p:
                out_regs.append({"pattern": p, "replacement": repl})
        if not out_reps and not out_regs:
            return defaults
        return {"replacements": out_reps, "regex_rules": out_regs}
    except FileNotFoundError:
        return defaults
    except Exception:
        return defaults


def _clamp_deg(n: float) -> float:
    v = float(n)
    return max(-float(MAX_DELTA_DEG), min(float(MAX_DELTA_DEG), v))


def parse_cn_degree_token(token: str) -> float | None:
    """Parse Arabic float/int or Chinese 0–99 style token (no 百)."""
    t = token.strip().replace(" ", "")
    if not t:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return float(t)
    t = t.replace("廿", "二十").replace("卅", "三十")
    if t == "十":
        return 10.0
    if t.startswith("十"):
        rest = t[1:]
        if not rest:
            return 10.0
        if rest not in _CN_DIGIT:
            return None
        return 10.0 + float(_CN_DIGIT[rest])
    if "十" in t:
        left, _, right = t.partition("十")
        tens_mul = _CN_DIGIT[left] if left else 1
        ones = _CN_DIGIT[right] if right else 0
        return float(tens_mul * 10 + ones)
    if len(t) == 1 and t in _CN_DIGIT:
        return float(_CN_DIGIT[t])
    return None


# Direction token + degree number (Arabic or Chinese).
#
# Important: allow light filler tokens between direction and 转/轉, e.g.
#   "向右再转三度" / "向右再轉三度"
_MOTION_DEG_RE = re.compile(
    r"(?:向(?P<dir_lrud>左|右|上|下)|(?P<dir_lr>左|右)(?:转|轉))\s*"
    r"(?:(?:再|再向|请|請)\s*)?"
    r"(?:(?:转|轉)\s*)?"
    r"(?P<deg>[\d零一二两三四五六七八九十廿卅]+|\d+(?:\.\d+)?)\s*度",
    re.UNICODE,
)


def _motion_raw_commands(clause: str) -> list[dict[str, Any]]:
    """Zero or more pan_by / tilt_by from explicit 左/右/上/下 … 度 spans."""
    out: list[dict[str, Any]] = []
    for m in _MOTION_DEG_RE.finditer(clause):
        tok = m.group("dir_lrud") or m.group("dir_lr")
        deg_tok = m.group("deg")
        if re.fullmatch(r"-?\d+(\.\d+)?", deg_tok):
            deg = float(deg_tok)
        else:
            pv = parse_cn_degree_token(deg_tok)
            if pv is None:
                continue
            deg = pv
        deg = _clamp_deg(deg)
        if tok == "左":
            out.append({"cmd": "pan_by", "d_pan_deg": _clamp_deg(-abs(deg)), "source": "voice"})
        elif tok == "右":
            out.append({"cmd": "pan_by", "d_pan_deg": _clamp_deg(abs(deg)), "source": "voice"})
        elif tok == "上":
            out.append({"cmd": "tilt_by", "d_tilt_deg": _clamp_deg(abs(deg)), "source": "voice"})
        elif tok == "下":
            out.append({"cmd": "tilt_by", "d_tilt_deg": _clamp_deg(-abs(deg)), "source": "voice"})
    return out


def _split_clauses(text: str) -> list[str]:
    t = text.strip()
    if not t:
        return []
    parts = re.split(
        # Note: do NOT split on bare \"再\" — it appears inside natural commands like
        # \"向右再转三度\"; we rely on stronger separators like \"然后/接着/之后\".
        r"(?:然后|然後|接着|接著|之后|之後|完了之後|完了之后|接下来|接下來|，|,|;|；|。|．)",
        t,
    )
    return [p.strip() for p in parts if p.strip()]


def _has_home(clause: str) -> bool:
    c = clause.lower()
    return bool(
        re.search(
            r"(回家|回初始|回原点|回\s*home|^home$|go\s*home|回\s*中位|复位|復位|reset\s*pose)",
            c,
            re.I,
        )
    )


def _has_search(clause: str) -> bool:
    c = clause.lower()
    return bool(
        re.search(
            r"(找人|搜寻|搜索|扫描|找一下|找\s*人|physical\s*search|^search$|find\s+(people|others|targets))",
            c,
            re.I,
        )
    )


def _pan_sign(clause: str) -> int | None:
    if re.search(r"(左|左转|左轉|向左|往左|逆时针|\bleft\b)", clause, re.I):
        if re.search(r"(右|右转|右轉|向右|往右|顺时针|\bright\b)", clause, re.I):
            return None
        return -1
    if re.search(r"(右|右转|右轉|向右|往右|顺时针|\bright\b)", clause, re.I):
        return 1
    return None


def _tilt_sign(clause: str) -> int | None:
    if re.search(r"(上|抬头|向上|朝上|pitch\s*up|tilt\s*up)", clause, re.I):
        if re.search(r"(下|低头|向下|朝下|pitch\s*down|tilt\s*down)", clause, re.I):
            return None
        return 1
    if re.search(r"(下|低头|向下|朝下|pitch\s*down|tilt\s*down)", clause, re.I):
        return -1
    return None


def _first_degree_value(clause: str) -> float | None:
    m = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:度|deg(?:rees)?\b)|"
        r"([-+]?\d+(?:\.\d+)?)(?=\s*°)|"
        r"([\d零一二两三四五六七八九十廿卅]+)\s*度",
        clause,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    if m.group(1) is not None:
        return float(m.group(1))
    if m.group(2) is not None:
        return float(m.group(2))
    cn = m.group(3)
    if cn is None:
        return None
    return parse_cn_degree_token(cn)


def _english_pan_tilt_degrees(clause: str) -> tuple[float | None, float | None]:
    dp = dt = None
    m = re.search(
        r"pan\s*([-+]?\d+(?:\.\d+)?)\s*(?:deg(?:rees)?|度)?",
        clause,
        re.I,
    )
    if m:
        dp = float(m.group(1))
    m = re.search(
        r"tilt\s*([-+]?\d+(?:\.\d+)?)\s*(?:deg(?:rees)?|度)?",
        clause,
        re.I,
    )
    if m:
        dt = float(m.group(1))
    return dp, dt


def _english_move_by(clause: str) -> dict[str, Any] | None:
    m = re.search(
        r"move_by.*?pan\s*([-+]?\d+(?:\.\d+)?).*?tilt\s*([-+]?\d+(?:\.\d+)?)",
        clause,
        re.I | re.DOTALL,
    )
    if not m:
        return None
    return {
        "cmd": "move_by",
        "d_pan_deg": _clamp_deg(float(m.group(1))),
        "d_tilt_deg": _clamp_deg(float(m.group(2))),
        "source": "voice",
    }


def _vague_pan_delta(clause: str) -> float | None:
    if not re.search(r"(一点|稍微|一点点|一點點|一點)", clause):
        return None
    base = 5.0
    if re.search(r"(左|左转|左轉|向左|往左)", clause):
        return -base
    if re.search(r"(右|右转|右轉|向右|往右)", clause):
        return base
    return base


def _parse_clause_raw_commands(clause: str) -> list[dict[str, Any]]:
    """Return 0..n **pre-validation** command dicts for one clause."""
    c = clause.strip()
    if not c:
        return []

    out: list[dict[str, Any]] = []
    # Allow compound clauses like: "先扫描再向右转十五度".
    if _has_home(c):
        out.append({"cmd": "home", "source": "voice"})
    if _has_search(c):
        out.append({"cmd": "search", "source": "voice"})

    motions = _motion_raw_commands(c)
    if motions:
        out.extend(motions)
        return out

    mb = _english_move_by(c)
    if mb is not None:
        out.append(mb)
        return out

    dp_en, dt_en = _english_pan_tilt_degrees(c)
    if dp_en is not None or dt_en is not None:
        raw: dict[str, Any] = {"cmd": "move_by", "source": "voice"}
        if dp_en is not None:
            raw["d_pan_deg"] = _clamp_deg(dp_en)
        if dt_en is not None:
            raw["d_tilt_deg"] = _clamp_deg(dt_en)
        out.append(raw)
        return out

    ps = _pan_sign(c)
    ts = _tilt_sign(c)
    deg = _first_degree_value(c)
    vague = _vague_pan_delta(c)

    if ps is not None and deg is not None and ts is None:
        out.append({"cmd": "pan_by", "d_pan_deg": _clamp_deg(ps * deg), "source": "voice"})
        return out
    if ts is not None and deg is not None and ps is None:
        out.append({"cmd": "tilt_by", "d_tilt_deg": _clamp_deg(ts * deg), "source": "voice"})
        return out

    if vague is not None:
        out.append({"cmd": "pan_by", "d_pan_deg": _clamp_deg(vague), "source": "voice"})
        return out

    return out


def _intent_has_non_noop(commands: tuple[dict[str, Any], ...]) -> bool:
    return any(c.get("cmd") != "noop" for c in commands)


def _intent_from_rules(text: str) -> IntentParseResult:
    """Rule layer only (no LLM)."""
    text = normalize_zh_command_text(text)
    clauses = _split_clauses(text)
    if not clauses:
        v = validate_voice_command({"cmd": "noop", "reason": "empty text"})
        assert v.normalized is not None
        return IntentParseResult((v.normalized,), "none", "empty")

    out: list[dict[str, Any]] = []
    any_rule = False
    for clause in clauses:
        raws = _parse_clause_raw_commands(clause)
        if not raws:
            continue
        any_rule = True
        for raw in raws:
            val = validate_voice_command(raw)
            if val.ok and val.normalized is not None:
                out.append(val.normalized)

    if not out:
        v = validate_voice_command({"cmd": "noop", "reason": "no matching rule"})
        assert v.normalized is not None
        return IntentParseResult((v.normalized,), "none", "no valid command after rules")

    return IntentParseResult(tuple(out), "rule" if any_rule else "none", None)


def _should_call_llm_fallback(
    *,
    use_llm: bool,
    llm_client: LlmIntentClient | None,
    text: str,
    rules: IntentParseResult,
    asr_low_confidence: bool,
    llm_only_on_asr_low_confidence: bool,
) -> bool:
    """
    Call LLM only when rules produced **no** actionable (non-noop) command.

    If ``llm_only_on_asr_low_confidence`` is True, also require ``asr_low_confidence``.
    """
    if not use_llm or llm_client is None:
        return False
    if not text.strip():
        return False
    if _intent_has_non_noop(rules.commands):
        return False
    if llm_only_on_asr_low_confidence:
        return bool(asr_low_confidence)
    return True


def _try_llm_command(
    user_prompt: str,
    llm_client: LlmIntentClient,
) -> IntentParseResult | None:
    raw = llm_client.propose_command(user_prompt)
    if raw is None:
        return None
    val = validate_voice_command(normalize_llm_command(raw))
    if not val.ok or val.normalized is None:
        return None
    if val.normalized.get("cmd") == "noop":
        return None
    return IntentParseResult((val.normalized,), "llm", None)


def parse_text_to_commands(
    text: str,
    *,
    use_llm: bool = False,
    llm_client: LlmIntentClient | None = None,
    llm_textfix_client: LlmTextFixClient | None = None,
    llm_bundle_client: LlmBundleClient | None = None,
    asr_low_confidence: bool = False,
    llm_only_on_asr_low_confidence: bool = False,
) -> IntentParseResult:
    """
    Rule-first intent parse, optional **local LLM** fallback.

    LLM runs **only** when rules yield no non-``noop`` command (rules always win).
    Set ``llm_only_on_asr_low_confidence=True`` to restrict LLM to cases where
    ``asr_low_confidence`` is also True (caller-reported ASR uncertainty).
    """
    text = normalize_zh_command_text(text)
    rules = _intent_from_rules(text)

    # Stage A (single call): bundle LLM can both fix transcript and propose a command.
    if (
        use_llm
        and llm_bundle_client is not None
        and not _intent_has_non_noop(rules.commands)
        and (not llm_only_on_asr_low_confidence or bool(asr_low_confidence))
        and text.strip()
    ):
        b = llm_bundle_client.propose_bundle(text)
        fixed = normalize_zh_command_text((b or {}).get("fixed_text", "") or "")
        cmd = (b or {}).get("command")
        if isinstance(cmd, dict):
            v = validate_voice_command(normalize_llm_command(cmd))
            if v.ok and v.normalized is not None and v.normalized.get("cmd") != "noop":
                return IntentParseResult((v.normalized,), "llm", "llm_bundle")
        if fixed.strip() and fixed.strip() != text.strip():
            fixed_rules = _intent_from_rules(fixed)
            if _intent_has_non_noop(fixed_rules.commands):
                return IntentParseResult(fixed_rules.commands, "rule", "llm_bundle_textfix")

    # Back-compat: textfix-only stage (older clients).
    if (
        use_llm
        and llm_bundle_client is None
        and llm_textfix_client is not None
        and not _intent_has_non_noop(rules.commands)
        and (not llm_only_on_asr_low_confidence or bool(asr_low_confidence))
        and text.strip()
    ):
        fixed = llm_textfix_client.propose_fixed_text(text)
        fixed = normalize_zh_command_text(fixed or "")
        if fixed.strip() and fixed.strip() != text.strip():
            fixed_rules = _intent_from_rules(fixed)
            if _intent_has_non_noop(fixed_rules.commands):
                return IntentParseResult(
                    fixed_rules.commands, fixed_rules.parser, "llm_textfix"
                )

    if not _should_call_llm_fallback(
        use_llm=use_llm,
        llm_client=llm_client,
        text=text,
        rules=rules,
        asr_low_confidence=asr_low_confidence,
        llm_only_on_asr_low_confidence=llm_only_on_asr_low_confidence,
    ):
        return rules

    assert llm_client is not None
    llm_res = _try_llm_command(text, llm_client)
    if llm_res is not None:
        note = "asr_low_confidence" if asr_low_confidence else None
        return IntentParseResult(llm_res.commands, "llm", note)
    # LLM was allowed and attempted, but produced nothing schema-valid.
    return IntentParseResult(rules.commands, rules.parser, "llm_failed")


def parse_after_clarification(
    original_transcript: str,
    followup_transcript: str,
    *,
    use_llm: bool = False,
    llm_client: LlmIntentClient | None = None,
    llm_textfix_client: LlmTextFixClient | None = None,
    llm_bundle_client: LlmBundleClient | None = None,
    asr_low_confidence: bool = False,
    llm_only_on_asr_low_confidence: bool = False,
) -> IntentParseResult:
    """
    Second turn after a fixed **one-round** clarification: rules on the follow-up
    first; if still no actionable command, call LLM once with **both** transcripts
    in the user prompt (rules still win if the follow-up alone is already clear).
    """
    fu = normalize_zh_command_text(followup_transcript).strip()
    if not fu:
        v = validate_voice_command({"cmd": "noop", "reason": "empty followup"})
        return IntentParseResult((v.normalized,), "none", "empty followup")

    follow_rules = _intent_from_rules(fu)
    if _intent_has_non_noop(follow_rules.commands):
        return follow_rules

    if (
        use_llm
        and llm_bundle_client is not None
        and (not llm_only_on_asr_low_confidence or bool(asr_low_confidence))
        and fu.strip()
    ):
        b = llm_bundle_client.propose_bundle(fu)
        fixed_fu = normalize_zh_command_text((b or {}).get("fixed_text", "") or "").strip()
        cmd = (b or {}).get("command")
        if isinstance(cmd, dict):
            v = validate_voice_command(normalize_llm_command(cmd))
            if v.ok and v.normalized is not None and v.normalized.get("cmd") != "noop":
                return IntentParseResult((v.normalized,), "llm", "after_clarify_llm_bundle")
        if fixed_fu and fixed_fu != fu:
            fixed_follow_rules = _intent_from_rules(fixed_fu)
            if _intent_has_non_noop(fixed_follow_rules.commands):
                return IntentParseResult(
                    fixed_follow_rules.commands, "rule", "after_clarify_llm_bundle_textfix"
                )

    if (
        use_llm
        and llm_textfix_client is not None
        and (not llm_only_on_asr_low_confidence or bool(asr_low_confidence))
        and fu.strip()
    ):
        fixed_fu = llm_textfix_client.propose_fixed_text(fu)
        fixed_fu = normalize_zh_command_text(fixed_fu or "").strip()
        if fixed_fu and fixed_fu != fu:
            fixed_follow_rules = _intent_from_rules(fixed_fu)
            if _intent_has_non_noop(fixed_follow_rules.commands):
                return IntentParseResult(
                    fixed_follow_rules.commands,
                    fixed_follow_rules.parser,
                    "after_clarify_llm_textfix",
                )

    if not _should_call_llm_fallback(
        use_llm=use_llm,
        llm_client=llm_client,
        text=fu,
        rules=follow_rules,
        asr_low_confidence=asr_low_confidence,
        llm_only_on_asr_low_confidence=llm_only_on_asr_low_confidence,
    ):
        return follow_rules

    assert llm_client is not None
    merged = (
        f"用户先说：{normalize_zh_command_text(original_transcript).strip()}\n"
        f"用户后说：{fu}\n"
        "请根据两句合并意图，只输出**一条**合法 JSON（cmd ∈ home, search, noop, move_by, pan_by, tilt_by；"
        "角度单位度，|Δ|≤60）。不要解释，不要 Markdown。"
    )
    llm_res = _try_llm_command(merged, llm_client)
    if llm_res is not None:
        return IntentParseResult(llm_res.commands, "llm", "after_clarify_llm")
    return IntentParseResult(follow_rules.commands, follow_rules.parser, "after_clarify_llm_failed")


def parse_text_to_command(
    text: str,
    *,
    use_llm: bool = False,
    llm_client: LlmIntentClient | None = None,
    llm_textfix_client: LlmTextFixClient | None = None,
    llm_bundle_client: LlmBundleClient | None = None,
    asr_low_confidence: bool = False,
    llm_only_on_asr_low_confidence: bool = False,
) -> dict[str, Any] | None:
    """
    First non-noop command if any; otherwise ``None`` when the only result is a single noop.
    Use ``parse_text_to_commands`` for the full validated list (including noop).
    """
    r = parse_text_to_commands(
        text,
        use_llm=use_llm,
        llm_client=llm_client,
        llm_textfix_client=llm_textfix_client,
        llm_bundle_client=llm_bundle_client,
        asr_low_confidence=asr_low_confidence,
        llm_only_on_asr_low_confidence=llm_only_on_asr_low_confidence,
    )
    if not r.commands:
        return None
    for cmd in r.commands:
        if cmd.get("cmd") != "noop":
            return cmd
    return None
