"""
Face detection + enrolled identity (LBPH).

Registry: `face_registry/<display_name>/*.jpg` — folder name is the HUD label.

LBPH `predict` returns a *distance* (not probability): **lower = better match**.
`match_threshold`: if distance > threshold, show "?" (unknown).
Requires `opencv-contrib-python` (`cv2.face`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.perception.base import PerceptionDetector
from src.perception._drawing import draw_label_bottom_center_in_box


def _require_cv2_face() -> None:
    if hasattr(cv2, "face"):
        return
    raise ImportError(
        "LBPH face recognition needs OpenCV *contrib* (module cv2.face).\n"
        "Your environment has a build without cv2.face (often `opencv-python`).\n\n"
        "Fix (pick one environment: venv or global):\n"
        "  pip uninstall opencv-python opencv-contrib-python -y\n"
        "  pip install opencv-contrib-python>=4.8\n\n"
        "Then verify:  python -c \"import cv2; print(hasattr(cv2, 'face'))\"  → True"
    )


def _bbox_to_aruco_style_corners(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array(
        [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
        dtype=np.float32,
    )


class FaceDetector(PerceptionDetector):
    """
    Haar frontal face + LBPH. Primary target = **largest** face bbox (Week 6.0).
    """

    def __init__(
        self,
        registry_root: str,
        match_threshold: float = 85.0,
        min_face_size: Tuple[int, int] = (80, 80),
        scale_factor: float = 1.08,
        min_neighbors: int = 5,
    ):
        root = Path(registry_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Face registry not found: {root}")

        self.registry_root = root
        self.match_threshold = float(match_threshold)
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

        _require_cv2_face()
        self._names: List[str] = []
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._train_from_registry()

        self.primary_display_name: Optional[str] = None
        self.primary_confidence: Optional[float] = None
        self._viz_faces: List[Tuple[int, int, int, int, str, float]] = []

        print(f"Face perception: {len(self._names)} enrolled — {', '.join(self._names)}")

    def _train_from_registry(self) -> None:
        faces: List[np.ndarray] = []
        labels: List[int] = []

        subdirs = sorted([p for p in self.registry_root.iterdir() if p.is_dir()])
        if not subdirs:
            raise ValueError(
                f"No person folders under {self.registry_root}. "
                "Create one subfolder per person with sample images (jpg/png)."
            )

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        per_person_samples: List[int] = []
        for label_idx, person_dir in enumerate(subdirs):
            name = person_dir.name
            self._names.append(name)
            image_files = sorted(
                f for f in person_dir.iterdir() if f.suffix.lower() in exts
            )
            if not image_files:
                raise ValueError(f"No images in {person_dir}")

            n_ok = 0
            for img_path in image_files:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    print(f"[warn] skip unreadable image: {img_path}")
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                rois = self._cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_face_size,
                )
                if len(rois) == 0:
                    print(f"[warn] no face in enrollment image, skip: {img_path}")
                    continue
                x, y, w, h = max(rois, key=lambda r: r[2] * r[3])
                roi = gray[y : y + h, x : x + w]
                faces.append(roi)
                labels.append(label_idx)
                n_ok += 1
            per_person_samples.append(n_ok)

        if any(n == 0 for n in per_person_samples):
            raise ValueError(
                "Some enrolled persons have no usable face crops. "
                "Add frontal face photos per folder."
            )

        self._recognizer.train(faces, np.array(labels, dtype=np.int32))

    def _predict_roi(self, gray_roi: np.ndarray) -> Tuple[int, float]:
        label, distance = self._recognizer.predict(gray_roi)
        return int(label), float(distance)

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Any]:
        self.primary_display_name = None
        self.primary_confidence = None
        self._viz_faces = []

        if frame is None:
            return False, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
        )

        if faces is None or len(faces) == 0:
            return False, None, None

        scored = []
        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            label, dist = self._predict_roi(roi)
            if dist <= self.match_threshold:
                name = self._names[label]
            else:
                name = "?"
            scored.append((x, y, w, h, name, dist, label))

        self._viz_faces = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in scored]
        primary = max(scored, key=lambda t: t[2] * t[3])
        px, py, pw, ph, pname, pdist, plabel = primary
        self.primary_display_name = pname
        self.primary_confidence = pdist

        corners = [_bbox_to_aruco_style_corners(px, py, pw, ph)]
        ids = np.array([[plabel]], dtype=np.int32)

        return True, ids, corners

    def visualize(
        self, frame: np.ndarray, corners: Any, ids: Optional[np.ndarray]
    ) -> np.ndarray:
        out = frame.copy()
        for (x, y, w, h, name, dist) in self._viz_faces:
            color = (0, 255, 0) if name != "?" else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"{name}"
            if name != "?":
                label += f" (d={dist:.0f})"
            draw_label_bottom_center_in_box(out, label, (x, y, w, h), color)

        if self._viz_faces:
            n = len(self._viz_faces)
            cv2.putText(
                out,
                f"Faces: {n}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
        return out
