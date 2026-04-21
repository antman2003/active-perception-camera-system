"""
Capture face enrollment images from the webcam into a named subfolder.

Examples:
  python scripts/capture_face_samples.py --registry face_registry --name 爸爸 --cam 1
  python scripts/capture_face_samples.py --registry face_registry --name Aaron Xie --cam 1

  (Names with spaces: pass as multiple tokens after --name, or quote in the shell:
   PowerShell: --name "Aaron Xie"   CMD/bash: same.)

Press SPACE to save the current frame (only if a frontal face is detected).
Press Q to quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.camera import open_videocapture, probe_camera_indices


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture face samples for LBPH registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tip (PowerShell): a space in the name must be quoted OR use multiple words:\n"
            '  --name "Aaron Xie"\n'
            "  --name Aaron Xie\n"
            "Do not use angle brackets <...> (PowerShell treats < as redirection).\n"
            "USB webcam on Windows: this script uses the same backends as src/camera (DSHOW/MSMF).\n"
            "If unsure which index is external:  python scripts/capture_face_samples.py --list-cams"
        ),
    )
    p.add_argument(
        "--list-cams",
        action="store_true",
        help="Try indices 0–9, print those that open and return a frame; then exit.",
    )
    p.add_argument(
        "--registry",
        type=str,
        default="face_registry",
        help="Root folder (person folder will be created inside).",
    )
    p.add_argument(
        "--name",
        nargs="*",
        default=None,
        metavar="WORD",
        help="Person folder / HUD label (required unless --list-cams). Example: --name Aaron Xie",
    )
    p.add_argument(
        "--cam",
        type=int,
        default=1,
        help="Camera index (0=often built-in, 1=often USB; use --list-cams to see which work)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=30,
        help="Stop after this many saved images (default 30)",
    )
    ns = p.parse_args()
    if ns.list_cams:
        return ns
    if not ns.name:
        p.error("--name is required unless you pass --list-cams")
    ns.display_name = " ".join(ns.name)
    return ns


def main():
    args = parse_args()
    if args.list_cams:
        found = probe_camera_indices()
        print("Working camera indices (opened + first frame OK):", found if found else "(none)")
        print("Use e.g.  --cam 1  with your external device if it appears above.")
        return

    out_dir = Path(args.registry) / args.display_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = open_videocapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open camera index {args.cam}. "
            f"Run: python scripts/capture_face_samples.py --list-cams"
        )

    idx = 0
    print(f"Saving to {out_dir.resolve()}")
    print("SPACE = save (face required), Q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.08, 5, minSize=(80, 80))
        vis = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"saved={idx}/{args.max}  SPACE|Q",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("capture_face_samples", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(faces) > 0 and idx < args.max:
            path = out_dir / f"sample_{idx:03d}.jpg"
            cv2.imwrite(str(path), frame)
            idx += 1
            print(f"  wrote {path.name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
