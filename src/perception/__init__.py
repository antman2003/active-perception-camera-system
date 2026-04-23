"""
Pluggable perception backends (Session 25).

- `ArucoDetector` — markers
- `FaceDetector` — Haar + LBPH + registry
- `CombinedPerception` — ArUco + face; one active primary per frame
"""

from __future__ import annotations

from src.face_registry_resolve import resolve_face_registry_dir
from src.perception.aruco import ArucoDetector
from src.perception.base import PerceptionDetector
from src.perception.combined import CombinedPerception
from src.perception.face import FaceDetector
from src.perception.hand import HandGestureDetector

# Backward-compatible name used in older snippets and demos.
PerceptionSystem = ArucoDetector


def create_perception(
    mode: str,
    *,
    face_registry_dir: str | None = None,
    face_match_threshold: float = 85.0,
    primary_hysteresis_frames: int = 0,
    mixed_policy: str = "aruco_first",
    marker_dict_id=None,
) -> PerceptionDetector:
    """
    Factory for `ActivePerceptionLoop`. Keeps construction out of the loop body.

    Args:
        mode: ``"aruco"`` | ``"face"`` | ``"mixed"`` | ``"auto"`` (alias of ``mixed``)
        face_registry_dir: For ``face`` / ``mixed``, registry root; ``None``/empty →
            ``<repo>/face_registry`` via ``resolve_face_registry_dir``.
        face_match_threshold: LBPH distance ceiling (lower distance = better match).
        primary_hysteresis_frames: Face / mixed face branch; two-face hysteresis (see ``FaceDetector``).
        mixed_policy: ``aruco_first`` | ``face_first`` | ``larger_area`` (``mixed`` / ``auto`` only).
        marker_dict_id: OpenCV ArUco dict id; default ``DICT_6X6_250``.
    """
    m = (mode or "aruco").lower().strip()
    if m == "auto":
        m = "mixed"
    if m == "aruco":
        kwargs = {}
        if marker_dict_id is not None:
            kwargs["marker_dict_id"] = marker_dict_id
        return ArucoDetector(**kwargs)
    if m == "face":
        face_registry_dir = resolve_face_registry_dir(face_registry_dir)
        return FaceDetector(
            face_registry_dir,
            match_threshold=face_match_threshold,
            primary_hysteresis_frames=primary_hysteresis_frames,
        )
    if m == "mixed":
        face_registry_dir = resolve_face_registry_dir(face_registry_dir)
        return CombinedPerception(
            mixed_policy=mixed_policy,
            face_registry_dir=face_registry_dir,
            face_match_threshold=face_match_threshold,
            primary_hysteresis_frames=primary_hysteresis_frames,
        )
    raise ValueError("perception mode must be 'aruco', 'face', 'mixed', or 'auto'")


__all__ = [
    "ArucoDetector",
    "CombinedPerception",
    "FaceDetector",
    "HandGestureDetector",
    "PerceptionDetector",
    "PerceptionSystem",
    "create_perception",
]
