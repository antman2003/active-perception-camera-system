"""
MediaPipe Hands + lightweight rule gestures (Session 27 / 27b).

Labels: ``thumbs_up``, ``heart`` (two hands), ``victory`` (✌),
``fist``, ``pointing``, or ``None``.

**Backends**: older wheels expose ``mediapipe.solutions.hands``; Python 3.12+ / 3.13
pip builds often ship **Tasks API only** (no ``solutions``). We use
``HandLandmarker`` + a one-time model download in that case.
"""

from __future__ import annotations

import math
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _require_mediapipe():
    try:
        import mediapipe as mp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Hand gestures need the `mediapipe` package.\n"
            "  pip install mediapipe\n"
        ) from e


def _hand_landmarker_model_path() -> str:
    """Return path to ``hand_landmarker.task``; download once into user cache."""
    cache = Path.home() / ".cache" / "active_perception_demo" / "hand_landmarker.task"
    if cache.exists() and cache.stat().st_size > 1_000_000:
        return str(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".task.part")
    try:
        urllib.request.urlretrieve(_HAND_LANDMARKER_URL, tmp)  # noqa: S310
    except Exception as e:
        raise RuntimeError(
            "Could not download MediaPipe hand landmarker model (first run needs network).\n"
            f"  URL: {_HAND_LANDMARKER_URL}\n"
            f"  Target: {cache}\n"
            "  Or download the file manually and place it at that path."
        ) from e
    tmp.replace(cache)
    return str(cache)


def _dist(lm, i: int, j: int) -> float:
    a, b = lm[i], lm[j]
    return float(math.hypot(a.x - b.x, a.y - b.y))


def _finger_extended(lm, tip: int, pip: int, mcp: int) -> bool:
    return _dist(lm, 0, tip) > _dist(lm, 0, pip) * 1.12


def _thumb_extended(lm) -> bool:
    return _dist(lm, 0, 4) > _dist(lm, 0, 3) * 1.08


def _two_hands_heart(lm_a, lm_b) -> bool:
    """Two thumbs + two index fingertips form the top of a heart (normalized coords)."""
    d_thumb = math.hypot(lm_a[4].x - lm_b[4].x, lm_a[4].y - lm_b[4].y)
    d_idx = math.hypot(lm_a[8].x - lm_b[8].x, lm_a[8].y - lm_b[8].y)
    d_wrist = math.hypot(lm_a[0].x - lm_b[0].x, lm_a[0].y - lm_b[0].y)
    if d_wrist < 0.04 or d_wrist > 0.55:
        return False
    return d_thumb < 0.13 and d_idx < 0.15 and d_thumb < 0.38 * d_wrist and d_idx < 0.42 * d_wrist


def _victory_v_sign(lm) -> bool:
    """Single-hand ✌: index+middle extended; ring+pinky+thumb not extended."""
    idx = _finger_extended(lm, 8, 6, 5)
    mid = _finger_extended(lm, 12, 10, 9)
    ring = _finger_extended(lm, 16, 14, 13)
    pink = _finger_extended(lm, 20, 18, 17)
    th = _thumb_extended(lm)
    if not (idx and mid):
        return False
    if ring or pink:
        return False
    if th:
        return False
    return True


def _classify_from_landmarks(lm) -> Optional[str]:
    idx = _finger_extended(lm, 8, 6, 5)
    mid = _finger_extended(lm, 12, 10, 9)
    ring = _finger_extended(lm, 16, 14, 13)
    pink = _finger_extended(lm, 20, 18, 17)
    th = _thumb_extended(lm)

    if th and not idx and not mid and not ring and not pink:
        return "thumbs_up"
    if _victory_v_sign(lm):
        return "victory"
    if idx and not mid and not ring and not pink:
        return "pointing"
    if not idx and not mid and not ring and not pink and not th:
        return "fist"
    return None


def hand_landmarks_to_aruco_corners(
    lm: Sequence[Any],
    frame_shape: Tuple[int, ...],
    pad_frac: float = 0.03,
) -> np.ndarray:
    """
    Tight axis-aligned bbox around one hand (normalized landmarks) → ArUco-style
    ``corners[0]`` for centroid / uncertainty / visual servo (pixel coords).
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    xs = [float(pt.x) * w for pt in lm]
    ys = [float(pt.y) * h for pt in lm]
    m = float(min(w, h))
    pad = pad_frac * m
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(w - 1), x1)
    y1 = min(float(h - 1), y1)
    xi, yi = int(x0), int(y0)
    wi = max(1, int(x1 - xi))
    hi = max(1, int(y1 - yi))
    return np.array(
        [[[xi, yi], [xi + wi, yi], [xi + wi, yi + hi], [xi, yi + hi]]],
        dtype=np.float32,
    )


def _classify_landmark_lists(
    lists: List[Sequence[Any]],
    handedness_scores: Optional[List[float]],
) -> Tuple[Optional[str], float]:
    """Shared rules for both ``solutions`` and Tasks backends."""
    if not lists:
        return None, 0.0
    if len(lists) >= 2:
        la, lb = lists[0], lists[1]
        if _two_hands_heart(la, lb):
            conf = 0.78
            if handedness_scores and len(handedness_scores) >= 2:
                conf = (handedness_scores[0] + handedness_scores[1]) / 2.0
            return "heart", conf
    lm = lists[0]
    label = _classify_from_landmarks(lm)
    conf = 0.75
    if handedness_scores:
        try:
            conf = float(handedness_scores[0])
        except Exception:
            pass
    return label, conf


class HandGestureDetector:
    """Runs MediaPipe Hands once per frame; returns a coarse gesture label."""

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.65):
        _require_mediapipe()
        import mediapipe as mp

        self._mp = mp
        self._backend: str
        self._hands = None
        self._task_det = None
        self._last_ts_ms = 0
        self.hands_visible: bool = False
        self._primary_hand_lm: Optional[Sequence[Any]] = None

        if hasattr(mp, "solutions") and getattr(mp.solutions, "hands", None) is not None:
            self._backend = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        else:
            self._backend = "tasks"
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _hand_landmarker_model_path()
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._task_det = mp_vision.HandLandmarker.create_from_options(opts)

    def _bump_task_timestamp_ms(self) -> int:
        t = int(time.monotonic() * 1000)
        if t <= self._last_ts_ms:
            t = self._last_ts_ms + 1
        self._last_ts_ms = t
        return t

    def close(self) -> None:
        try:
            if self._backend == "solutions" and self._hands is not None:
                self._hands.close()
            elif self._backend == "tasks" and self._task_det is not None:
                self._task_det.close()
        except Exception:
            pass

    @property
    def primary_hand_landmarks(self) -> Optional[Sequence[Any]]:
        """First detected hand (normalized landmarks), or ``None``."""
        return self._primary_hand_lm

    def classify(self, frame: Any) -> Tuple[Optional[str], float]:
        """
        Returns:
            (gesture_name or None, detection confidence 0..1 from MediaPipe handedness / presence).
        """
        if frame is None or frame.size == 0:
            self.hands_visible = False
            self._primary_hand_lm = None
            return None, 0.0
        self.hands_visible = False
        self._primary_hand_lm = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            assert self._hands is not None
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None, 0.0
            lists = [h.landmark for h in res.multi_hand_landmarks]
            scores = None
            if res.multi_handedness:
                scores = []
                for mh in res.multi_handedness:
                    try:
                        scores.append(float(mh.classification[0].score))
                    except Exception:
                        scores.append(0.75)
        else:
            assert self._task_det is not None
            from mediapipe.tasks.python.vision.core import image as mp_image

            arr = np.ascontiguousarray(rgb)
            mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, arr)
            ts = self._bump_task_timestamp_ms()
            tres = self._task_det.detect_for_video(mp_img, ts)
            if not tres.hand_landmarks:
                return None, 0.0
            lists = list(tres.hand_landmarks)
            scores = []
            for block in tres.handedness:
                if block:
                    scores.append(float(block[0].score))
                else:
                    scores.append(0.75)

        self.hands_visible = len(lists) > 0
        self._primary_hand_lm = lists[0] if lists else None

        label, conf = _classify_landmark_lists(lists, scores)
        return label, conf

    def draw_debug(self, frame: Any) -> Any:
        """Optional: draw hand skeleton on a copy (for HUD)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = frame.copy()
        h, w = frame.shape[:2]

        if self._backend == "solutions":
            assert self._hands is not None
            res = self._hands.process(rgb)
            if not res.multi_hand_landmarks:
                return out
            for hand_lm in res.multi_hand_landmarks:
                for p in hand_lm.landmark:
                    cx, cy = int(p.x * w), int(p.y * h)
                    cv2.circle(out, (cx, cy), 2, (200, 200, 255), -1)
            return out

        assert self._task_det is not None
        from mediapipe.tasks.python.vision.core import image as mp_image

        arr = np.ascontiguousarray(rgb)
        mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, arr)
        ts = self._bump_task_timestamp_ms()
        tres = self._task_det.detect_for_video(mp_img, ts)
        if not tres.hand_landmarks:
            return out
        for lm_row in tres.hand_landmarks:
            for p in lm_row:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(out, (cx, cy), 2, (200, 200, 255), -1)
        return out
