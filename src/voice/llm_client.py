"""
Optional local LLM client for voice intent (Session 30 step 3 fallback).

Uses stdlib HTTP only (no extra dependency). Default backend: Ollama ``/api/chat``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import time
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LlmIntentClient(Protocol):
    """Given user transcript, return one command **dict** (pre-schema) or ``None``."""

    def propose_command(self, user_text: str) -> dict[str, Any] | None: ...


@runtime_checkable
class LlmTextFixClient(Protocol):
    """
    Given user transcript, return a *corrected* transcript string, or ``None``.

    This is a safety layer: it must NOT output commands; we re-run rules after fixing.
    """

    def propose_fixed_text(self, user_text: str) -> str | None: ...


@runtime_checkable
class LlmBundleClient(Protocol):
    """
    Single-call bundle: both corrected text and a proposed schema command.
    """

    def propose_bundle(self, user_text: str) -> dict[str, Any] | None: ...


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_llm_command_json(content: str) -> dict[str, Any] | None:
    """Parse assistant message into a single JSON object, or ``None``."""
    s = _strip_json_fence(content)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        i = s.find("{")
        j = s.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(s[i : j + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def parse_llm_textfix_json(content: str) -> str | None:
    """
    Parse assistant message into {"fixed_text": "..."} and return the string.
    """
    obj = parse_llm_command_json(content)
    if not isinstance(obj, dict):
        return None
    ft = obj.get("fixed_text")
    if not isinstance(ft, str):
        return None
    ft = ft.strip()
    return ft or None


def parse_llm_bundle_json(content: str) -> dict[str, Any] | None:
    """
    Parse assistant message into {"fixed_text": "...", "command": {...}}.

    `command` may be missing or null; caller validates it against schema.
    """
    obj = parse_llm_command_json(content)
    if not isinstance(obj, dict):
        return None
    out: dict[str, Any] = {}
    ft = obj.get("fixed_text")
    if isinstance(ft, str) and ft.strip():
        out["fixed_text"] = ft.strip()
    cmd = obj.get("command")
    if isinstance(cmd, dict):
        out["command"] = cmd
    return out if out else None


def normalize_llm_command(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize common LLM output variants into our frozen schema fields.

    Our schema (see `src/voice_intent/schema.py`) expects:
      - pan_by:  {"cmd":"pan_by",  "d_pan_deg": <number>}
      - tilt_by: {"cmd":"tilt_by", "d_tilt_deg": <number>}
      - move_by: {"cmd":"move_by", "d_pan_deg": <number?>, "d_tilt_deg": <number?>}

    Some models tend to return alternatives like:
      - delta_pan_deg / delta_tilt_deg
      - delta: {pan_deg, tilt_deg}
      - delta: {pan, tilt}
      - pan_deg / tilt_deg (top-level)
      - pan / tilt (top-level)

    This function is intentionally conservative: it only renames fields; it does not
    invent values or change cmd names.
    """
    out: dict[str, Any] = dict(obj)

    # Unwrap nested delta dict if present.
    delta = out.get("delta")
    if isinstance(delta, dict):
        # Only lift if the target fields aren't already present.
        if "d_pan_deg" not in out:
            for k in ("pan_deg", "pan", "delta_pan_deg", "d_pan_deg"):
                if k in delta:
                    out["d_pan_deg"] = delta[k]
                    break
        if "d_tilt_deg" not in out:
            for k in ("tilt_deg", "tilt", "delta_tilt_deg", "d_tilt_deg"):
                if k in delta:
                    out["d_tilt_deg"] = delta[k]
                    break

    # Map common top-level aliases.
    if "d_pan_deg" not in out:
        for k in ("delta_pan_deg", "pan_deg", "pan"):
            if k in out:
                out["d_pan_deg"] = out[k]
                break
    if "d_tilt_deg" not in out:
        for k in ("delta_tilt_deg", "tilt_deg", "tilt"):
            if k in out:
                out["d_tilt_deg"] = out[k]
                break

    # Drop a nested delta if we lifted values; otherwise keep it for debug.
    if "delta" in out and isinstance(out.get("delta"), dict) and (
        "d_pan_deg" in out or "d_tilt_deg" in out
    ):
        out.pop("delta", None)

    # Remove alias keys to satisfy schema strictness (unknown keys are rejected).
    for k in ("delta_pan_deg", "pan_deg", "pan", "delta_tilt_deg", "tilt_deg", "tilt"):
        if k in out and k not in ("d_pan_deg", "d_tilt_deg"):
            out.pop(k, None)

    return out


class OllamaIntentClient:
    """Call Ollama ``POST /api/chat`` (local)."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def propose_command(self, user_text: str) -> dict[str, Any] | None:
        system = (
            "You map a short spoken instruction for a pan-tilt camera into ONE JSON object.\n"
            "Allowed cmd values only: home, search, noop, move_by, pan_by, tilt_by.\n"
            "Semantics: d_pan_deg left negative, right positive; d_tilt_deg up positive, down negative; "
            f"each |delta| <= 60.\n"
            "If you cannot map safely, output {\"cmd\":\"noop\",\"reason\":\"llm_unsure\"}.\n"
            "Output ONLY valid JSON, one object, no markdown, no explanation."
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text.strip()},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return None
        except json.JSONDecodeError:
            return None
        finally:
            _ = t0  # reserved for future timing hook
        msg = payload.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            return None
        return parse_llm_command_json(content)

    def propose_fixed_text(self, user_text: str) -> str | None:
        system = (
            "You correct ASR transcription errors for a pan-tilt camera voice command.\n"
            "Task: rewrite the user's text into a short, clearer command sentence in Chinese or English.\n"
            "Rules:\n"
            "- You MUST NOT output any command JSON.\n"
            "- You MUST output ONLY valid JSON: {\"fixed_text\":\"...\"}\n"
            "- If no correction is needed, set fixed_text to the original text.\n"
            "- If you are unsure, keep fixed_text as the original text.\n"
            "No markdown. No explanation."
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text.strip()},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return None
        except json.JSONDecodeError:
            return None
        msg = payload.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            return None
        return parse_llm_textfix_json(content)

    def propose_bundle(self, user_text: str) -> dict[str, Any] | None:
        system = (
            "You help map a spoken instruction for a pan-tilt camera.\n"
            "You must output ONLY valid JSON with this shape:\n"
            "{\"fixed_text\":\"...\",\"command\":{...}}\n"
            "\n"
            "Rules:\n"
            "- fixed_text: rewrite the user's text into a short, clearer command sentence (keep original language if possible).\n"
            "- command: ONE schema command object with cmd ∈ home, search, noop, move_by, pan_by, tilt_by.\n"
            "- Use fields d_pan_deg / d_tilt_deg (left negative, right positive; up positive, down negative), |delta|<=60.\n"
            "- Axis safety: if fixed_text contains 上/下 (or 'tilt up/down'), you MUST use tilt_by (no pan_by / move_by).\n"
            "- Axis safety: if fixed_text contains 左/右 (or 'turn left/right'/'pan left/right'), you MUST use pan_by (no tilt_by / move_by).\n"
            "- If fixed_text mentions ONLY one axis, do NOT output move_by.\n"
            "- If you cannot map safely, set command to {\"cmd\":\"noop\",\"reason\":\"llm_unsure\"}.\n"
            "- No markdown. No explanation. JSON only."
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text.strip()},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return None
        except json.JSONDecodeError:
            return None
        msg = payload.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            return None
        return parse_llm_bundle_json(content)


class MockLlmIntentClient:
    """Deterministic dict for unit tests."""

    def __init__(self, payload: dict[str, Any] | None):
        self._payload = payload

    def propose_command(self, user_text: str) -> dict[str, Any] | None:
        if self._payload is None:
            return None
        return dict(self._payload)


class MockLlmTextFixClient:
    """Deterministic string for unit tests."""

    def __init__(self, fixed_text: str | None):
        self._fixed_text = fixed_text

    def propose_fixed_text(self, user_text: str) -> str | None:
        if self._fixed_text is None:
            return None
        return str(self._fixed_text)


class MockLlmBundleClient:
    def __init__(self, payload: dict[str, Any] | None):
        self._payload = payload

    def propose_bundle(self, user_text: str) -> dict[str, Any] | None:
        if self._payload is None:
            return None
        return dict(self._payload)
