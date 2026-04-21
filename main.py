"""
Unified CLI entry point for the active perception system.
"""

import argparse
import sys

from demo import run_full_demo, probe_pan_tilt
from src.benchmark import run_benchmark
from src.policy import run_policy_demo
from src.uncertainty import run_uncertainty_demo


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Active Perception System CLI")
    parser.add_argument(
        "--mode",
        choices=["full", "uncertainty", "policy", "benchmark"],
        default="full",
        help="Run mode: full system, uncertainty inspection, policy test, or benchmark.",
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
        help="Enable debug mode. Affects frame logging in full and benchmark modes.",
    )
    parser.add_argument(
        "--system",
        choices=["all", "static", "active_exp", "active_full"],
        default="all",
        help="Benchmark system variant to run. Only used in benchmark mode.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Benchmark duration per run in seconds. Only used in benchmark mode.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label to attach to benchmark results.",
    )
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=None,
        help="Optional manual distance annotation for benchmark results.",
    )
    parser.add_argument(
        "--lux",
        type=float,
        default=None,
        help="Optional manual lux annotation for benchmark results.",
    )
    parser.add_argument(
        "--no-pan-tilt",
        action="store_true",
        help="Force disable pan-tilt stage (skip auto-detection).",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port for the pan-tilt Arduino (default: COM3).",
    )
    return parser.parse_args(argv)


def print_mode_banner(mode: str, cam_id: int, debug: bool,
                      pan_tilt: bool = False, port: str = "COM3"):
    print("=" * 60)
    print("    Active Perception Camera System")
    print("=" * 60)
    print(f"[*] Mode: {mode}")
    print(f"[*] Camera: {cam_id}")
    print(f"[*] Debug: {'ON' if debug else 'OFF'}")
    print(f"[*] Pan-Tilt: {'ENABLED on ' + port if pan_tilt else 'DISABLED'}")
    print("=" * 60)


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

    print_mode_banner(args.mode, args.cam, args.debug, use_pan_tilt, args.port)

    try:
        if args.mode == "full":
            run_full_demo(
                camera_id=args.cam,
                debug=args.debug,
                enable_pan_tilt=use_pan_tilt,
                pan_tilt_port=args.port,
            )
        elif args.mode == "uncertainty":
            if args.debug:
                print("[i] --debug is ignored in uncertainty mode.")
            run_uncertainty_demo(camera_id=args.cam)
        elif args.mode == "policy":
            if args.debug:
                print("[i] --debug is ignored in policy mode.")
            run_policy_demo(camera_id=args.cam)
        elif args.mode == "benchmark":
            run_benchmark(
                camera_id=args.cam,
                duration_s=args.duration,
                system=args.system,
                debug=args.debug,
                label=args.label,
                distance_cm=args.distance_cm,
                lux=args.lux,
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
