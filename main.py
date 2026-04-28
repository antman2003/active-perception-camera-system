"""
Unified CLI entry point for the active perception system.
"""

import argparse
import sys

from demo import apply_privacy_cli_preset, run_full_demo, probe_pan_tilt
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
    parser.add_argument(
        "--perception",
        choices=["aruco", "face", "mixed", "auto"],
        default="mixed",
        help="full mode: mixed (default) | auto (=mixed) | aruco | face.",
    )
    parser.add_argument(
        "--mixed-policy",
        choices=["aruco_first", "face_first", "larger_area"],
        default="aruco_first",
        help="With mixed|auto: which target drives tracking (default: aruco_first).",
    )
    parser.add_argument(
        "--face-registry",
        type=str,
        default=None,
        help="Enrolled faces root. Omitted in face|mixed|auto → repo ./face_registry (must exist).",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=85.0,
        help="LBPH distance threshold for face ID (default: 85).",
    )
    parser.add_argument(
        "--no-auto-exposure",
        action="store_true",
        help="Disable automatic exposure sweeps (full mode).",
    )
    parser.add_argument(
        "--face-primary-hysteresis",
        type=int,
        default=0,
        metavar="N",
        help="Face mode: defer primary switch when two faces are similar size (0=off).",
    )
    parser.add_argument(
        "--gesture-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Session 27b: MediaPipe hand gestures (default: on). Disable with --no-gesture-actions.",
    )
    parser.add_argument(
        "--privacy",
        action="store_true",
        help="One-flag strong display privacy (blur + labels). See demo --help.",
    )
    # Session 30: voice (PTT via 'v' in the OpenCV window)
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice thread (PTT: press 'v' in the video window).",
    )
    parser.add_argument(
        "--voice-mode",
        type=str,
        default="ptt",
        choices=("ptt", "always"),
        help="Voice mode: ptt (press 'v') or always (wake word + VAD capture). Default: ptt.",
    )
    parser.add_argument(
        "--voice-lang",
        type=str,
        default="zh",
        help="ASR language hint: zh | en | auto (default: zh).",
    )
    parser.add_argument(
        "--voice-model",
        type=str,
        default="base",
        help="faster-whisper model size for live PTT (default: base).",
    )
    parser.add_argument(
        "--voice-record-seconds",
        type=float,
        default=5.0,
        help="PTT recording window seconds per 'v' press (default: 5.0).",
    )
    parser.add_argument(
        "--voice-save-wav",
        action="store_true",
        help="Save each PTT recording as a .wav under logs/blackbox/<session>/audio/ (debugging).",
    )
    parser.add_argument(
        "--voice-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable local LLM fallback (Ollama) when rules yield noop (default: on). Disable with --no-voice-llm.",
    )
    parser.add_argument(
        "--voice-llm-model",
        type=str,
        default="qwen2.5:1.5b",
        help="Ollama model name for fallback (default: qwen2.5:1.5b).",
    )
    parser.add_argument(
        "--voice-clarify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable one-round clarification prompt when rules yield noop (default: off).",
    )
    parser.add_argument(
        "--privacy-blur-faces",
        action="store_true",
        help="Face blur with default tuning; use --privacy for stronger one-liner.",
    )
    parser.add_argument(
        "--privacy-blur-kernel",
        type=int,
        default=99,
        help="Blur kernel (odd, >=3; default 99).",
    )
    parser.add_argument(
        "--privacy-blur-pad",
        type=float,
        default=0.10,
        metavar="R",
        help="BBox pad ratio before blur (default 0.10).",
    )
    parser.add_argument(
        "--privacy-blur-passes",
        type=int,
        default=2,
        metavar="N",
        help="Blur passes per ROI (default 2).",
    )
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    apply_privacy_cli_preset(args, raw)
    return args


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
                perception_mode=args.perception,
                face_registry_dir=args.face_registry,
                face_match_threshold=args.face_threshold,
                enable_exposure_control=not args.no_auto_exposure,
                primary_hysteresis_frames=args.face_primary_hysteresis,
                mixed_policy=args.mixed_policy,
                enable_gesture_actions=args.gesture_actions,
                privacy_blur_faces=args.privacy_blur_faces,
                privacy_blur_kernel=args.privacy_blur_kernel,
                privacy_blur_pad=args.privacy_blur_pad,
                privacy_blur_passes=args.privacy_blur_passes,
                enable_voice=args.voice,
                voice_mode=args.voice_mode,
                voice_lang=args.voice_lang,
                voice_model=args.voice_model,
                voice_record_seconds=args.voice_record_seconds,
                voice_llm=args.voice_llm,
                voice_llm_model=args.voice_llm_model,
                voice_clarify=args.voice_clarify,
                voice_save_wav=args.voice_save_wav,
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
