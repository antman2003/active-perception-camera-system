"""
Unified CLI entry point for the active perception system.
"""

import argparse
import sys

from demo import run_full_demo
from src.policy import run_policy_demo
from src.uncertainty import run_uncertainty_demo


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Active Perception System CLI")
    parser.add_argument(
        "--mode",
        choices=["full", "uncertainty", "policy"],
        default="full",
        help="Run mode: full system, uncertainty inspection, or policy test.",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=1,
        help="Camera device index (0 for built-in laptop cam, 1 for external USB cam. Default: 1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Currently only affects full mode frame logging.",
    )
    return parser.parse_args(argv)


def print_mode_banner(mode: str, cam_id: int, debug: bool):
    print("=" * 60)
    print("    Active Perception Camera System")
    print("=" * 60)
    print(f"[*] Mode: {mode}")
    print(f"[*] Camera: {cam_id}")
    print(f"[*] Debug: {'ON' if debug else 'OFF'}")
    print("=" * 60)


def main(argv=None):
    args = parse_args(argv)
    print_mode_banner(args.mode, args.cam, args.debug)

    try:
        if args.mode == "full":
            run_full_demo(camera_id=args.cam, debug=args.debug)
        elif args.mode == "uncertainty":
            if args.debug:
                print("[i] --debug is ignored in uncertainty mode.")
            run_uncertainty_demo(camera_id=args.cam)
        elif args.mode == "policy":
            if args.debug:
                print("[i] --debug is ignored in policy mode.")
            run_policy_demo(camera_id=args.cam)
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
