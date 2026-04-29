"""
Draw Unicode text on OpenCV BGR frames (Windows-friendly).

OpenCV's built-in Hershey fonts do not support Chinese, so the HUD needs Pillow.
This helper tries common Windows CJK fonts and falls back gracefully.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

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


def unicode_font_available(font_size: int = 18) -> bool:
    """
    Returns True if we can load a TTF font for Unicode HUD text.
    """
    try:
        return _load_font(int(font_size)) is not None
    except Exception:
        return False


def try_draw_text_bgr(
    frame_bgr: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    font_size: int = 18,
    color_bgr: tuple[int, int, int] = (255, 220, 180),
) -> bool:
    """
    Best-effort Unicode text draw. Returns True if we believe drawing happened.

    We avoid full-frame comparisons; we compare only the ROI that would be affected.
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        return False
    s = (text or "").strip()
    if not s:
        return False

    font = _load_font(font_size)
    if font is None:
        return False

    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    fill_rgb = (r, g, b)

    tmp = Image.new("RGB", (4, 4), (0, 0, 0))
    d_tmp = ImageDraw.Draw(tmp)
    try:
        bbox = d_tmp.textbbox((0, 0), s, font=font)
        left, top, right, bot = bbox
    except Exception:
        tw, th = d_tmp.textsize(s, font=font)
        left, top, right, bot = 0, 0, tw, th

    pad = 6
    w = int(math.ceil(right - left)) + pad * 2
    h = int(math.ceil(bot - top)) + pad * 2
    if w <= 0 or h <= 0:
        return False

    fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    x0 = int(x) + int(math.floor(left)) - pad
    y0 = int(y) + int(math.floor(top)) - pad
    x1 = x0 + w
    y1 = y0 + h

    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = min(fw, x1)
    fy1 = min(fh, y1)
    if fx0 >= fx1 or fy0 >= fy1:
        return False

    before = frame_bgr[fy0:fy1, fx0:fx1].copy()

    roi_rgb = before[:, :, ::-1]
    img = Image.fromarray(roi_rgb)
    draw = ImageDraw.Draw(img)
    px = int(x) - fx0 + pad - int(math.floor(left))
    py = int(y) - fy0 + pad - int(math.floor(top))
    draw.text((px, py), s, font=font, fill=fill_rgb)

    roi_out_rgb = np.asarray(img)
    after = roi_out_rgb[:, :, ::-1]
    frame_bgr[fy0:fy1, fx0:fx1] = after

    try:
        return not np.array_equal(before, after)
    except Exception:
        return True


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

    Implementation note:
    - Historically this converted the *entire* frame BGR<->RGB for Pillow.
    - For realtime HUD overlays, that's far too expensive; we only rasterize into a tight ROI.
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        return frame_bgr
    s = (text or "").strip()
    if not s:
        return frame_bgr

    font = _load_font(font_size)
    if font is None:
        return frame_bgr

    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    fill_rgb = (r, g, b)

    tmp = Image.new("RGB", (4, 4), (0, 0, 0))
    d_tmp = ImageDraw.Draw(tmp)
    try:
        bbox = d_tmp.textbbox((0, 0), s, font=font)
        left, top, right, bot = bbox
    except Exception:
        # Pillow compatibility fallback.
        tw, th = d_tmp.textsize(s, font=font)
        left, top, right, bot = 0, 0, tw, th

    pad = 6
    w = int(math.ceil(right - left)) + pad * 2
    h = int(math.ceil(bot - top)) + pad * 2
    if w <= 0 or h <= 0:
        return frame_bgr

    fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    x0 = int(x) + int(math.floor(left)) - pad
    y0 = int(y) + int(math.floor(top)) - pad
    x1 = x0 + w
    y1 = y0 + h

    # Clamp ROI to frame
    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = min(fw, x1)
    fy1 = min(fh, y1)
    if fx0 >= fx1 or fy0 >= fy1:
        return frame_bgr

    roi_bgr = frame_bgr[fy0:fy1, fx0:fx1].copy()

    roi_rgb = roi_bgr[:, :, ::-1]
    img = Image.fromarray(roi_rgb)
    draw = ImageDraw.Draw(img)
    px = int(x) - fx0 + pad - int(math.floor(left))
    py = int(y) - fy0 + pad - int(math.floor(top))
    draw.text((px, py), s, font=font, fill=fill_rgb)

    roi_out_rgb = np.asarray(img)
    frame_bgr[fy0:fy1, fx0:fx1] = roi_out_rgb[:, :, ::-1]
    return frame_bgr