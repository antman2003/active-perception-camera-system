"""
Build a small GIF from screenshots (for social posts).

Examples (PowerShell, from repo root):

  # Two-frame blink: before / after (same canvas size after resize)
  python scripts/make_gif.py blink `
    "logs/blackbox/20260419_202143/screenshots/sniper_timeout_f845.png" `
    "logs/blackbox/20260419_202143/screenshots/sniper_lock_f541.png" `
    -o publication/sniper_timeout_vs_lock.gif --frame-ms 700

  # Concat left | right (single static wide frame — not animated; use blink for motion)
  python scripts/make_gif.py side-by-side img_a.png img_b.png -o publication/compare.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _load_rgb(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im


def cmd_blink(args: argparse.Namespace) -> None:
    a, b = Path(args.a), Path(args.b)
    if not a.is_file() or not b.is_file():
        raise SystemExit(f"Missing image: {a} or {b}")

    im1 = _load_rgb(a)
    im2 = _load_rgb(b)
    if im2.size != im1.size:
        im2 = im2.resize(im1.size, Image.Resampling.LANCZOS)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = int(args.frame_ms)
    im1.save(
        out,
        save_all=True,
        append_images=[im2],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out.resolve()}  (2 frames, {duration} ms each, loop)")


def cmd_side_by_side(args: argparse.Namespace) -> None:
    a, b = Path(args.a), Path(args.b)
    im1 = _load_rgb(a)
    im2 = _load_rgb(b)
    h = max(im1.height, im2.height)
    w1 = int(im1.width * h / im1.height)
    w2 = int(im2.width * h / im2.height)
    im1 = im1.resize((w1, h), Image.Resampling.LANCZOS)
    im2 = im2.resize((w2, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w1 + w2 + int(args.gap), h), (32, 32, 32))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (w1 + int(args.gap), 0))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        canvas.save(out, quality=92)
    else:
        canvas.save(out)
    print(f"Wrote {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make GIF / compare image from screenshots.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_blink = sub.add_parser("blink", help="Two-image blink GIF (A B A B ...)")
    p_blink.add_argument("a", help="First image path")
    p_blink.add_argument("b", help="Second image path")
    p_blink.add_argument("-o", "--output", required=True, help="Output .gif path")
    p_blink.add_argument("--frame-ms", type=int, default=600, help="Display time per frame (ms)")
    p_blink.set_defaults(func=cmd_blink)

    p_side = sub.add_parser("side-by-side", help="Horizontal concat (PNG/JPEG, not GIF)")
    p_side.add_argument("a")
    p_side.add_argument("b")
    p_side.add_argument("-o", "--output", required=True)
    p_side.add_argument("--gap", type=int, default=8, help="Pixels between images")
    p_side.set_defaults(func=cmd_side_by_side)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
