"""
One-round clarification for voice intent (Session 30 extension).

First transcript: **rules only** (no LLM). If no actionable command and
``enable_clarify`` is on, return a fixed Chinese prompt — **no** servo write.

Second transcript: rules on the follow-up; if still noop, **one** LLM call with
both lines in the prompt (same schema validation as everywhere else).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.voice.intent import (
    IntentParseResult,
    _intent_from_rules,
    _intent_has_non_noop,
    parse_after_clarification,
    parse_text_to_commands,
)
from src.voice.llm_client import LlmBundleClient, LlmIntentClient, LlmTextFixClient
from src.voice_intent.schema import validate_voice_command

DEFAULT_CLARIFY_PROMPT_ZH = (
    "没听清具体指令。请用短句再说一次：例如「回家」「找人扫描」，"
    "或「左转十度」「右转五度」「向上二十度」。"
)


Phase = Literal["resolved", "clarify"]


@dataclass(frozen=True)
class VoiceIntentClarifyResult:
    phase: Phase
    commands: tuple[dict[str, Any], ...]
    """Empty when ``phase == \"clarify\"`` (nothing to execute yet)."""

    parser: str
    """``rule`` | ``llm`` | ``none`` | ``clarify`` (prompt-only turn)."""

    clarify_prompt_zh: str | None
    note: str | None = None


def _wrap_resolved(r: IntentParseResult) -> VoiceIntentClarifyResult:
    return VoiceIntentClarifyResult(
        "resolved", r.commands, r.parser, None, r.note
    )


class VoiceIntentClarifySession:
    """
    Stateful **single** clarification round.

    Not thread-safe; use one instance per voice worker / user session.
    """

    def __init__(self) -> None:
        self._pending_original: str | None = None

    def reset(self) -> None:
        self._pending_original = None

    @property
    def awaiting_clarification(self) -> bool:
        return self._pending_original is not None

    def feed(
        self,
        text: str,
        *,
        use_llm: bool = False,
        llm_client: LlmIntentClient | None = None,
        llm_textfix_client: LlmTextFixClient | None = None,
        llm_bundle_client: LlmBundleClient | None = None,
        enable_clarify: bool = True,
        clarify_prompt_zh: str | None = None,
        asr_low_confidence: bool = False,
        llm_only_on_asr_low_confidence: bool = False,
    ) -> VoiceIntentClarifyResult:
        t = (text or "").strip()
        prompt = clarify_prompt_zh or DEFAULT_CLARIFY_PROMPT_ZH

        # --- follow-up turn (one round only) ---
        if self._pending_original is not None:
            orig = self._pending_original
            self._pending_original = None
            if not t:
                v = validate_voice_command(
                    {"cmd": "noop", "reason": "empty clarification reply"}
                )
                assert v.normalized is not None
                return VoiceIntentClarifyResult(
                    "resolved", (v.normalized,), "none", None, "aborted clarify"
                )
            r = parse_after_clarification(
                orig,
                t,
                use_llm=use_llm,
                llm_client=llm_client,
                llm_textfix_client=llm_textfix_client,
                llm_bundle_client=llm_bundle_client,
                asr_low_confidence=asr_low_confidence,
                llm_only_on_asr_low_confidence=llm_only_on_asr_low_confidence,
            )
            return _wrap_resolved(r)

        # --- first turn ---
        if not t:
            v = validate_voice_command({"cmd": "noop", "reason": "empty text"})
            assert v.normalized is not None
            return VoiceIntentClarifyResult(
                "resolved", (v.normalized,), "none", None, "empty"
            )

        if enable_clarify:
            rules = _intent_from_rules(t)
            if _intent_has_non_noop(rules.commands):
                return _wrap_resolved(rules)
            self._pending_original = t
            return VoiceIntentClarifyResult(
                "clarify", (), "clarify", prompt, "awaiting one follow-up transcript"
            )

        r = parse_text_to_commands(
            t,
            use_llm=use_llm,
            llm_client=llm_client,
            llm_textfix_client=llm_textfix_client,
            llm_bundle_client=llm_bundle_client,
            asr_low_confidence=asr_low_confidence,
            llm_only_on_asr_low_confidence=llm_only_on_asr_low_confidence,
        )
        return _wrap_resolved(r)
