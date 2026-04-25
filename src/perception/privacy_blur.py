"""
Demo / 录屏隐私：在已检测的人脸 bbox 内做高斯模糊（仅影响显示帧，不改变检测用原图）。

调用方应对 **visualize 之后的副本** 做模糊，避免污染闭环所用的 ``frame``。
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np


def collect_face_boxes(perception: Any) -> List[Tuple[int, int, int, int]]:
    """从 ``FaceDetector`` 或 ``CombinedPerception`` 取本帧人脸框；ArUco-only 返回空。"""
    face = getattr(perception, "_face", None)
    if face is not None:
        boxes = getattr(face, "last_face_boxes", None)
        return list(boxes) if boxes is not None else []
    boxes = getattr(perception, "last_face_boxes", None)
    return list(boxes) if boxes is not None else []


def blur_face_boxes_bgr(
    image: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    *,
    kernel_size: int = 99,
    pad_ratio: float = 0.10,
    passes: int = 2,
) -> np.ndarray:
    """
    在 ``image`` 上 **原地** 模糊每个 bbox（BGR）。返回 ``image`` 便于链式调用。

    ``kernel_size`` 会强制为奇数且 ≥3。``pad_ratio`` 在宽高上外扩再糊，减少贴边漏脸。
    ``passes``：同一 ROI 上连续高斯模糊次数（≥1），多遍叠加明显强于单遍大核。
    """
    if image is None or image.size == 0 or not boxes:
        return image
    h0, w0 = image.shape[:2]
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    n_pass = max(1, int(passes))
    pr = max(0.0, float(pad_ratio))
    for (x, y, w, h) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        px = max(0, int(pr * w)) if w > 0 else 0
        py = max(0, int(pr * h)) if h > 0 else 0
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w0, x + w + px)
        y2 = min(h0, y + h + py)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        for _ in range(n_pass):
            roi = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return image
