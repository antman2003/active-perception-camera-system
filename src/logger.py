"""
Lightweight blackbox logger for the active perception system.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


class BlackboxLogger:
    """
    Writes frame snapshots and event logs into a timestamped session folder.
    """

    def __init__(
        self,
        root_dir: str = "logs/blackbox",
        enabled: bool = True,
        frame_logging_enabled: bool = False,
    ):
        self.enabled = enabled
        self.frame_logging_enabled = frame_logging_enabled
        self.session_dir: Optional[Path] = None
        self.frames_path: Optional[Path] = None
        self.events_path: Optional[Path] = None
        self.screenshots_dir: Optional[Path] = None

        if not self.enabled:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(root_dir) / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.session_dir / "frames.jsonl"
        self.events_path = self.session_dir / "events.jsonl"
        self.screenshots_dir = self.session_dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    def log_frame(self, **payload) -> None:
        if not self.enabled or not self.frame_logging_enabled or self.frames_path is None:
            return
        self._append_jsonl(self.frames_path, payload)

    def log_event(self, event_type: str, frame=None, screenshot_name: Optional[str] = None, **payload) -> None:
        if not self.enabled or self.events_path is None:
            return

        event = {"event_type": event_type, **payload}
        if frame is not None and screenshot_name:
            screenshot_path = self.save_screenshot(frame, screenshot_name)
            if screenshot_path is not None:
                event["screenshot"] = screenshot_path

        self._append_jsonl(self.events_path, event)

    def save_screenshot(self, frame, name: str) -> Optional[str]:
        if not self.enabled or self.screenshots_dir is None or frame is None:
            return None

        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        filename = f"{safe_name}.png"
        path = self.screenshots_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            **payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
