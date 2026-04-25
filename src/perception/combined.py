"""
ArUco + face in one frame: both detectors run; one primary drives the control loop.

HUD draws both; ``active_backend`` is ``\"aruco\"`` | ``\"face\"`` | ``None``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.perception.aruco import ArucoDetector
from src.perception.base import PerceptionDetector
from src.perception.face import FaceDetector


def _primary_area(corners) -> float:
    if corners is None or len(corners) < 1:
        return 0.0
    try:
        return float(cv2.contourArea(corners[0].astype(np.float32)))
    except Exception:
        return 0.0


class CombinedPerception(PerceptionDetector):
    """
    ``mixed_policy``:
    - ``aruco_first`` — marker wins if present, else face (good for desk demos).
    - ``face_first`` — face wins if present, else marker.
    - ``larger_area`` — compare primary contour areas when both exist.
    """

    def __init__(
        self,
        *,
        mixed_policy: str = "aruco_first",
        face_registry_dir: str,
        face_match_threshold: float = 85.0,
        primary_hysteresis_frames: int = 0,
    ):
        pol = (mixed_policy or "aruco_first").lower().strip()
        if pol not in ("aruco_first", "face_first", "larger_area"):
            raise ValueError(
                "mixed_policy must be 'aruco_first', 'face_first', or 'larger_area'"
            )
        self.mixed_policy = pol
        self._aruco = ArucoDetector()
        self._face = FaceDetector(
            face_registry_dir,
            match_threshold=face_match_threshold,
            primary_hysteresis_frames=primary_hysteresis_frames,
        )
        self.active_backend: Optional[str] = None
        self._aruco_det = False
        self._face_det = False
        self._aruco_ids = None
        self._face_ids = None
        self._aruco_corners = None
        self._face_corners = None
        print(f"Perception (mixed): policy={self.mixed_policy}")

    @property
    def last_face_boxes(self) -> List[Tuple[int, int, int, int]]:
        return self._face.last_face_boxes

    def _pick(self) -> Tuple[bool, Any, Any]:
        a_ok = self._aruco_det
        f_ok = self._face_det

        if self.mixed_policy == "aruco_first":
            if a_ok:
                self.active_backend = "aruco"
                return True, self._aruco_ids, self._aruco_corners
            if f_ok:
                self.active_backend = "face"
                return True, self._face_ids, self._face_corners
            self.active_backend = None
            return False, None, None

        if self.mixed_policy == "face_first":
            if f_ok:
                self.active_backend = "face"
                return True, self._face_ids, self._face_corners
            if a_ok:
                self.active_backend = "aruco"
                return True, self._aruco_ids, self._aruco_corners
            self.active_backend = None
            return False, None, None

        # larger_area
        if a_ok and f_ok:
            aa = _primary_area(self._aruco_corners)
            fa = _primary_area(self._face_corners)
            if fa >= aa:
                self.active_backend = "face"
                return True, self._face_ids, self._face_corners
            self.active_backend = "aruco"
            return True, self._aruco_ids, self._aruco_corners
        if a_ok:
            self.active_backend = "aruco"
            return True, self._aruco_ids, self._aruco_corners
        if f_ok:
            self.active_backend = "face"
            return True, self._face_ids, self._face_corners
        self.active_backend = None
        return False, None, None

    def pick_aruco_before_face(self) -> Tuple[bool, Optional[np.ndarray], Any]:
        """
        Re-resolve primary target as **ArUco over face** (both already detected this frame).

        Used when ``--gesture-actions`` is on so overall priority is
        **hand > ArUco > face**; when no hand is visible, ArUco wins over face
        regardless of ``mixed_policy``.
        """
        a_ok = self._aruco_det
        f_ok = self._face_det
        if a_ok:
            self.active_backend = "aruco"
            return True, self._aruco_ids, self._aruco_corners
        if f_ok:
            self.active_backend = "face"
            return True, self._face_ids, self._face_corners
        self.active_backend = None
        return False, None, None

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Any]:
        self._aruco_det, self._aruco_ids, self._aruco_corners = self._aruco.detect(frame)
        self._face_det, self._face_ids, self._face_corners = self._face.detect(frame)
        return self._pick()

    def visualize(
        self,
        frame: np.ndarray,
        corners: Any,
        ids: Optional[np.ndarray],
        face_label_override: Optional[str] = None,
    ) -> np.ndarray:
        out = frame.copy()
        if self._aruco_det:
            out = self._aruco.visualize(
                out, self._aruco_corners, self._aruco_ids, face_label_override
            )
        if self._face_det:
            out = self._face.visualize(
                out, self._face_corners, self._face_ids, face_label_override
            )
        tag = self.active_backend or "none"
        # Baseline ~50: below "Faces:" / marker HUD at y≈30, above uncertainty bar in loop (y≈62+)
        cv2.putText(
            out,
            f"ACTIVE->{tag}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        return out
