"""
Draw Unicode text on OpenCV BGR frames (Windows-friendly).

OpenCV's built-in Hershey fonts do not support Chinese, so the HUD needs Pillow.
This helper tries common Windows CJK fonts and falls back gracefully.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_WINDOWS_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\msyh.ttc",  # Microsoft YaHei
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",  # SimHei
    r"C:\Windows\Fonts\simsun.ttc",  # SimSun
    r"C:\Windows\Fonts\arial.ttf",
)


@lru_cache(maxsize=16)
def _load_font(font_size: int, font_paths: tuple[str, ...] = _WINDOWS_FONT_CANDIDATES):
    size = max(10, int(font_size))
    for p in font_paths:
        try:
            if Path(p).is_file():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return None


def draw_text_bgr(
    frame_bgr: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    font_size: int = 18,
    color_bgr: tuple[int, int, int] = (255, 220, 180),
) -> np.ndarray:
    """
    Draw Unicode text onto a BGR OpenCV frame. Returns the modified frame.

    If no TTF font is available, returns the original frame unchanged.
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        return frame_bgr
    s = (text or "").strip()
    if not s:
        return frame_bgr

    font = _load_font(font_size)
    if font is None:
        return frame_bgr

    # BGR -> RGB for Pillow
    rgb = frame_bgr[:, :, ::-1]
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    draw.text((int(x), int(y)), s, font=font, fill=(r, g, b))
    out_rgb = np.asarray(img)
    # RGB -> BGR
    frame_bgr[:, :, :] = out_rgb[:, :, ::-1]
    return frame_bgr

