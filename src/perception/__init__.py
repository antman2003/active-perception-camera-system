"""
Pluggable perception backends (Session 25).

- `ArucoDetector` — markers
- `FaceDetector` — Haar + LBPH + registry
"""

from __future__ import annotations

from src.perception.aruco import ArucoDetector
from src.perception.base import PerceptionDetector
from src.perception.face import FaceDetector

# Backward-compatible name used in older snippets and demos.
PerceptionSystem = ArucoDetector


def create_perception(
    mode: str,
    *,
    face_registry_dir: str | None = None,
    face_match_threshold: float = 85.0,
    marker_dict_id=None,
) -> PerceptionDetector:
    """
    Factory for `ActivePerceptionLoop`. Keeps construction out of the loop body.

    Args:
        mode: ``"aruco"`` | ``"face"``
        face_registry_dir: Required when mode is ``"face"``.
        face_match_threshold: LBPH distance ceiling (lower distance = better match).
        marker_dict_id: OpenCV ArUco dict id; default ``DICT_6X6_250``.
    """
    m = (mode or "aruco").lower().strip()
    if m == "aruco":
        kwargs = {}
        if marker_dict_id is not None:
            kwargs["marker_dict_id"] = marker_dict_id
        return ArucoDetector(**kwargs)
    if m == "face":
        if not face_registry_dir:
            raise ValueError("perception_mode='face' requires face_registry_dir")
        return FaceDetector(
            face_registry_dir,
            match_threshold=face_match_threshold,
        )
    raise ValueError("perception mode must be 'aruco' or 'face'")


__all__ = [
    "ArucoDetector",
    "FaceDetector",
    "PerceptionDetector",
    "PerceptionSystem",
    "create_perception",
]
