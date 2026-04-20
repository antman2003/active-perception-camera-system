"""
Active Perception Camera System - Demo Entry Point

This script starts the complete active perception loop (Exposure + Zoom Control).
It handles command-line arguments to make it easy to switch cameras or settings
without modifying the core logic files.
"""

import argparse
import sys
import serial
from src.loop import ActivePerceptionLoop

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Active Perception System Demo")
    parser.add_argument(
        "--cam", 
        type=int, 
        default=1, 
        help="Camera device index (0 for built-in laptop cam, 1 for external USB cam. Default: 1)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with per-frame blackbox logging."
    )
    parser.add_argument(
        "--no-pan-tilt",
        action="store_true",
        help="Force disable pan-tilt stage (skip auto-detection)."
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port for the pan-tilt Arduino (default: COM3)."
    )
    return parser.parse_args(argv)


def probe_pan_tilt(port: str) -> bool:
    """Try to open the serial port briefly to check if the Arduino is connected."""
    try:
        s = serial.Serial(port, 115200, timeout=1)
        s.close()
        return True
    except (serial.SerialException, OSError):
        return False


def print_welcome_message(cam_id, debug, pan_tilt, port):
    print("="*60)
    print("    Active Perception Camera System")
    print("="*60)
    print(f"[*] Camera: {cam_id}")
    print(f"[*] Debug mode: {'ON' if debug else 'OFF'}")
    if pan_tilt:
        print(f"[*] Pan-Tilt: ENABLED on {port}")
    else:
        print("[*] Pan-Tilt: DISABLED")
    print("[*] Make sure you have a 6x6 ArUco Marker ready.")
    print("[*] Features Active:")
    print("    - Auto-Exposure Sweep (on lighting change)")
    print("    - Auto-Digital Zoom (on target distance change)")
    print("    - Sniper Recovery Mode (if target is lost while zoomed)")
    print("    - Visual Servoing (Smooth ROI Tracking)")
    if pan_tilt:
        print("    - Physical Pan-Tilt Tracking")
        print("    - Physical Spiral Search")
    print("    Press 'q' in the video window to quit.")
    print("="*60)


def run_full_demo(camera_id: int = 1, debug: bool = False,
                  enable_pan_tilt: bool = True, pan_tilt_port: str = "COM3"):
    print_welcome_message(camera_id, debug, enable_pan_tilt, pan_tilt_port)
    app = ActivePerceptionLoop(
        camera_id=camera_id,
        debug=debug,
        enable_pan_tilt=enable_pan_tilt,
        pan_tilt_port=pan_tilt_port,
    )
    app.run()

def main(argv=None):
    args = parse_args(argv)

    if args.no_pan_tilt:
        use_pan_tilt = False
    else:
        print(f"[*] Probing pan-tilt stage on {args.port}...")
        use_pan_tilt = probe_pan_tilt(args.port)
        if use_pan_tilt:
            print(f"[*] Pan-tilt stage detected on {args.port}.")
        else:
            print(f"[*] No pan-tilt stage found on {args.port}. Running in digital-only mode.")

    try:
        run_full_demo(
            camera_id=args.cam,
            debug=args.debug,
            enable_pan_tilt=use_pan_tilt,
            pan_tilt_port=args.port,
        )
    except RuntimeError as e:
        print(f"\n[ERROR] Failed to start system: {e}")
        print(f"        Is camera {args.cam} connected and not used by another program?")
        print("        Try running with '--cam 0' for your built-in webcam.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[i] Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
