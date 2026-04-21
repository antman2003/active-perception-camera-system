"""UTF-8 labels on BGR frames (Chinese names on Windows)."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None  # type: ignore


def draw_label(
    img: np.ndarray, text: str, origin: Tuple[int, int], bgr: Tuple[int, int, int]
) -> None:
    x, y = origin
    if Image is None or all(ord(c) < 128 for c in text):
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            bgr,
            2,
        )
        return

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_cn_font(22)
    rgb = (bgr[2], bgr[1], bgr[0])
    draw.text((x, y), text, font=font, fill=rgb)
    img[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_label_bottom_center_in_box(
    img: np.ndarray,
    text: str,
    box: Tuple[int, int, int, int],
    bgr: Tuple[int, int, int],
    margin: int = 4,
) -> None:
    """Draw ``text`` inside ``box`` (x, y, w, h), centered horizontally, along the inner bottom edge."""
    x, y, bw, bh = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if bw <= 0 or bh <= 0:
        return

    if Image is None or all(ord(c) < 128 for c in text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 2
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        if tw + 2 * margin <= bw:
            tx = x + (bw - tw) // 2
        else:
            tx = x + margin
        # putText y = baseline; keep ink inside box (descenders use ``bl`` below baseline)
        baseline_y = y + bh - margin - int(bl)
        if baseline_y - th < y + margin:
            baseline_y = y + margin + th
        cv2.putText(img, text, (tx, baseline_y), font, scale, bgr, thick, cv2.LINE_AA)
        return

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_cn_font(22)
    rgb = (bgr[2], bgr[1], bgr[0])
    try:
        left, top0, right, bottom = font.getbbox(text)
    except AttributeError:
        left, top0, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top0
    tx = x + max(margin, (bw - tw) // 2)
    tx = min(tx, x + bw - tw - margin)
    tx = max(x + margin, tx)
    ty = y + bh - th - margin
    ty = max(y + margin, min(ty, y + bh - th - margin))
    draw.text((tx, ty), text, font=font, fill=rgb)
    img[:, :, :] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _load_cn_font(size: int):
    if Image is None:
        return None
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simhei.ttf"),
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()
