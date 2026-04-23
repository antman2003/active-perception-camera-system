"""
Session 27b: map stable gestures to short pan-tilt choreography (opt-in).

Safety: confirm frames, cooldown, pattern lock.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Pattern step: (wait_frames, d_pan, d_tilt, settle_s). settle_s == _HOME_SENTINEL → home().
_HOME_SENTINEL = -1.0


class GestureActionEngine:
    """
    - ``fist`` (confirmed): **「锁定」短编排** — 三连快速 micro-tilt + 左右微 pan 收紧感 → ``home``。
    - ``thumbs_up`` (confirmed): **tilt ±20°** nod **×3** groups.
    - ``victory`` (confirmed): large tilt + pan celebration, then ``home``.
    - ``heart`` (confirmed): pan sway + tilt nod + ``home``。
    """

    def __init__(
        self,
        confirm_frames: int = 10,
        cooldown_frames: int = 48,
        victory_confirm_frames: int = 12,
        victory_cooldown_frames: int = 120,
        heart_confirm_frames: int = 12,
        heart_cooldown_frames: int = 90,
        fist_confirm_frames: int = 12,
        fist_cooldown_frames: int = 72,
    ):
        self.confirm_frames = max(3, int(confirm_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.victory_confirm_frames = max(6, int(victory_confirm_frames))
        self.victory_cooldown_frames = max(0, int(victory_cooldown_frames))
        self.heart_confirm_frames = max(6, int(heart_confirm_frames))
        self.heart_cooldown_frames = max(0, int(heart_cooldown_frames))
        self.fist_confirm_frames = max(6, int(fist_confirm_frames))
        self.fist_cooldown_frames = max(0, int(fist_cooldown_frames))

        self._cooldown = 0
        self._streak_label: Optional[str] = None
        self._streak = 0
        self._pattern: Optional[List[Tuple[int, float, float, float]]] = None
        self._pat_i = 0
        self._pat_wait = 0

    @property
    def pattern_active(self) -> bool:
        return self._pattern is not None

    @property
    def pt_suppress(self) -> bool:
        """Stop visual-servo pan-tilt nudge while executing a gesture pattern."""
        return self.pattern_active

    def tick(self, label: Optional[str], context) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1

        if self._pattern is not None:
            self._advance_pattern(context)
            return

        if label and label == self._streak_label:
            self._streak += 1
        else:
            self._streak_label = label
            self._streak = 1 if label else 0

        if self._cooldown > 0:
            return

        mon = (
            getattr(context, "enable_pan_tilt", False)
            and getattr(context, "pan_tilt", None) is not None
            and getattr(context, "current_state", None) is not None
            and context.current_state.name == "MONITOR"
        )

        if label == "heart" and self._streak >= self.heart_confirm_frames and mon:
            self._start_heart_celebration()
            self._cooldown = self.heart_cooldown_frames
            self._streak = 0
            self._streak_label = None
            log = getattr(context, "blackbox", None)
            if log is not None:
                log.log_event(
                    "gesture_action_heart",
                    frame_idx=context.frame_count,
                )
            return

        if label == "victory" and self._streak >= self.victory_confirm_frames and mon:
            self._start_victory_celebration()
            self._cooldown = self.victory_cooldown_frames
            self._streak = 0
            self._streak_label = None
            log = getattr(context, "blackbox", None)
            if log is not None:
                log.log_event(
                    "gesture_action_victory",
                    frame_idx=context.frame_count,
                )
            return

        if label == "fist" and self._streak >= self.fist_confirm_frames and mon:
            self._start_fist_lock_pose()
            self._cooldown = self.fist_cooldown_frames
            self._streak = 0
            self._streak_label = None
            log = getattr(context, "blackbox", None)
            if log is not None:
                log.log_event(
                    "gesture_action_fist",
                    frame_idx=context.frame_count,
                )
            return

        if label == "thumbs_up" and self._streak >= self.confirm_frames and mon:
            self._start_thumbs_nod()
            self._cooldown = self.cooldown_frames
            self._streak = 0
            self._streak_label = None
            log = getattr(context, "blackbox", None)
            if log is not None:
                log.log_event(
                    "gesture_action_thumbs_up",
                    frame_idx=context.frame_count,
                )

    def _start_thumbs_nod(self) -> None:
        steps: List[Tuple[int, float, float, float]] = []
        tilt = 20.0
        settle = 0.18
        for _ in range(3):
            steps.append((2, 0.0, tilt, settle))
            steps.append((2, 0.0, -tilt, settle))
        self._pattern = steps
        self._pat_i = 0
        self._pat_wait = steps[0][0]

    def _start_fist_lock_pose(self) -> None:
        """握拳：快速三连 micro-tilt + 左右微 pan（「收紧 / 锁定」感）→ home。"""
        settle_t = 0.12
        settle_p = 0.14
        steps: List[Tuple[int, float, float, float]] = []
        for _ in range(3):
            steps.append((2, 0.0, 6.0, settle_t))
            steps.append((2, 0.0, -6.0, settle_t))
        steps.append((2, 9.0, 0.0, settle_p))
        steps.append((2, -18.0, 0.0, settle_p))
        steps.append((2, 9.0, 0.0, settle_p))
        steps.append((5, 0.0, 0.0, _HOME_SENTINEL))
        self._pattern = steps
        self._pat_i = 0
        self._pat_wait = steps[0][0]

    def _start_heart_celebration(self) -> None:
        """P2: pan 小摆 2 周期 + tilt 点头 3 组 + home。"""
        wait = 3
        settle = 0.18
        pan_s = 14.0
        steps: List[Tuple[int, float, float, float]] = []
        for _ in range(2):
            steps.append((wait, pan_s, 0.0, settle))
            steps.append((wait, -pan_s, 0.0, settle))
        for _ in range(3):
            steps.append((2, 0.0, 7.0, 0.14))
            steps.append((2, 0.0, -7.0, 0.14))
        steps.append((6, 0.0, 0.0, _HOME_SENTINEL))
        self._pattern = steps
        self._pat_i = 0
        self._pat_wait = steps[0][0]

    def _start_victory_celebration(self) -> None:
        wait = 4
        settle = 0.22
        tilt_big = 26.0
        pan_big = 42.0
        steps: List[Tuple[int, float, float, float]] = []
        for _ in range(2):
            steps.append((wait, 0.0, tilt_big, settle))
            steps.append((wait, 0.0, -tilt_big, settle))
        for _ in range(2):
            steps.append((wait, pan_big, 0.0, settle))
            steps.append((wait, -pan_big, 0.0, settle))
        steps.append((8, 0.0, 0.0, _HOME_SENTINEL))
        self._pattern = steps
        self._pat_i = 0
        self._pat_wait = steps[0][0]

    def _advance_pattern(self, context) -> None:
        assert self._pattern is not None
        if self._pat_wait > 0:
            self._pat_wait -= 1
            return
        if self._pat_i >= len(self._pattern):
            self._pattern = None
            return
        _, d_pan, d_tilt, settle_s = self._pattern[self._pat_i]
        self._pat_i += 1
        try:
            if settle_s == _HOME_SENTINEL:
                context.pan_tilt.home(smooth=True, settle_s=0.45)
            else:
                context.pan_tilt.nudge(d_pan, d_tilt, settle_s=settle_s)
        except Exception:
            self._pattern = None
            return
        if self._pat_i < len(self._pattern):
            self._pat_wait = self._pattern[self._pat_i][0]
        else:
            self._pattern = None
