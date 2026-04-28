"""
Active Perception Camera System - Demo Entry Point

This script starts the complete active perception loop (Exposure + Zoom Control).
It handles command-line arguments to make it easy to switch cameras or settings
without modifying the core logic files.
"""

import argparse
import sys
import serial
from src.face_registry_resolve import resolve_face_registry_dir
from src.loop import ActivePerceptionLoop

# Strong display-privacy preset for ``--privacy`` (matches prior CLI recommendations).
PRIVACY_CLI_PRESET_KERNEL = 151
PRIVACY_CLI_PRESET_PASSES = 3
PRIVACY_CLI_PRESET_PAD = 0.15


def _argv_privacy_blur_tokens(argv: list[str]) -> dict[str, bool]:
    """True if argv contains an explicit ``--privacy-blur-*`` token (supports ``--opt=value``)."""
    out = {"kernel": False, "passes": False, "pad": False}
    for t in argv:
        if t.startswith("--privacy-blur-kernel"):
            out["kernel"] = True
        elif t.startswith("--privacy-blur-passes"):
            out["passes"] = True
        elif t.startswith("--privacy-blur-pad"):
            out["pad"] = True
    return out


def apply_privacy_cli_preset(args: argparse.Namespace, argv: list[str]) -> None:
    """
    If ``--privacy``: enable face blur and apply strong defaults, unless the user
    already passed the corresponding ``--privacy-blur-*`` flag.
    """
    if not getattr(args, "privacy", False):
        return
    args.privacy_blur_faces = True
    explicit = _argv_privacy_blur_tokens(argv)
    if not explicit["kernel"]:
        args.privacy_blur_kernel = PRIVACY_CLI_PRESET_KERNEL
    if not explicit["passes"]:
        args.privacy_blur_passes = PRIVACY_CLI_PRESET_PASSES
    if not explicit["pad"]:
        args.privacy_blur_pad = PRIVACY_CLI_PRESET_PAD


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
    parser.add_argument(
        "--perception",
        choices=["aruco", "face", "mixed", "auto"],
        default="mixed",
        help="mixed (default) | auto (=mixed) | aruco | face.",
    )
    parser.add_argument(
        "--mixed-policy",
        choices=["aruco_first", "face_first", "larger_area"],
        default="aruco_first",
        help="When --perception mixed|auto: which target drives tracking (default: aruco_first).",
    )
    parser.add_argument(
        "--face-registry",
        type=str,
        default=None,
        help="Root folder of enrolled faces (one subfolder per person). "
        "Omitted → use ./face_registry under the repo root (must exist).",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=85.0,
        help="LBPH match threshold (lower distance = more confident; default 85).",
    )
    parser.add_argument(
        "--no-auto-exposure",
        action="store_true",
        help="Disable automatic exposure sweeps (compare fixed exposure vs Session 26 face tuning).",
    )
    parser.add_argument(
        "--face-primary-hysteresis",
        type=int,
        default=0,
        metavar="N",
        help="Face mode: when two enrolled faces are similar size, switch primary only after N frames (0=off).",
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
        help="One-flag display privacy: blur faces + anonymized HUD labels, with strong "
        f"defaults (kernel {PRIVACY_CLI_PRESET_KERNEL}, {PRIVACY_CLI_PRESET_PASSES} passes, "
        f"pad {PRIVACY_CLI_PRESET_PAD}). Override pieces with --privacy-blur-kernel / "
        "--privacy-blur-passes / --privacy-blur-pad.",
    )
    parser.add_argument(
        "--privacy-blur-faces",
        action="store_true",
        help="Same blur pipeline as --privacy but uses milder built-in defaults unless you "
        "add --privacy-blur-kernel / --privacy-blur-passes / --privacy-blur-pad.",
    )
    parser.add_argument(
        "--privacy-blur-kernel",
        type=int,
        default=99,
        help="Blur kernel size (odd, >=3; default 99). Larger = stronger blur.",
    )
    parser.add_argument(
        "--privacy-blur-pad",
        type=float,
        default=0.10,
        metavar="R",
        help="Expand each face bbox by this fraction before blur (default 0.10).",
    )
    parser.add_argument(
        "--privacy-blur-passes",
        type=int,
        default=2,
        metavar="N",
        help="Gaussian blur passes per face ROI (default 2). More passes = stronger.",
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
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    apply_privacy_cli_preset(args, raw)
    return args


def probe_pan_tilt(port: str) -> bool:
    """Try to open the serial port briefly to check if the Arduino is connected."""
    try:
        s = serial.Serial(port, 115200, timeout=1)
        s.close()
        return True
    except (serial.SerialException, OSError):
        return False


def print_welcome_message(
    cam_id,
    debug,
    pan_tilt,
    port,
    perception: str,
    face_registry,
    enable_exposure_control: bool = True,
    mixed_policy: str | None = None,
):
    print("="*60)
    print("    Active Perception Camera System")
    print("="*60)
    print(f"[*] Camera: {cam_id}")
    print(f"[*] Perception: {perception.upper()}")
    if perception in ("face", "mixed"):
        print(f"[*] Face registry: {face_registry}")
    if perception == "mixed" and mixed_policy:
        print(f"[*] Mixed policy: {mixed_policy}")
    print(f"[*] Auto exposure sweeps: {'ON' if enable_exposure_control else 'OFF'}")
    print(f"[*] Debug mode: {'ON' if debug else 'OFF'}")
    if pan_tilt:
        print(f"[*] Pan-Tilt: ENABLED on {port}")
    else:
        print("[*] Pan-Tilt: DISABLED")
    if perception == "aruco":
        print("[*] Make sure you have a 6x6 ArUco Marker ready.")
    elif perception == "mixed":
        print("[*] Mixed mode: ArUco + face; HUD shows both; ACTIVE picks the tracker.")
    else:
        print("[*] Face mode: show enrolled people to the camera.")
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


def run_full_demo(
    camera_id: int = 1,
    debug: bool = False,
    enable_pan_tilt: bool = True,
    pan_tilt_port: str = "COM3",
    perception_mode: str = "aruco",
    face_registry_dir: str | None = None,
    face_match_threshold: float = 85.0,
    enable_exposure_control: bool = True,
    primary_hysteresis_frames: int = 0,
    mixed_policy: str = "aruco_first",
    enable_gesture_actions: bool = True,
    privacy_blur_faces: bool = False,
    privacy_blur_kernel: int = 99,
    privacy_blur_pad: float = 0.10,
    privacy_blur_passes: int = 2,
    enable_voice: bool = False,
    voice_mode: str = "ptt",
    voice_lang: str = "zh",
    voice_model: str = "base",
    voice_record_seconds: float = 5.0,
    voice_llm: bool = False,
    voice_llm_model: str = "qwen2.5:1.5b",
    voice_clarify: bool = True,
    voice_save_wav: bool = False,
):
    pm = (perception_mode or "aruco").lower().strip()
    if pm == "auto":
        pm = "mixed"
    perception_mode = pm
    resolved_fr = face_registry_dir or ""
    if perception_mode in ("face", "mixed"):
        resolved_fr = resolve_face_registry_dir(face_registry_dir)
    print_welcome_message(
        camera_id,
        debug,
        enable_pan_tilt,
        pan_tilt_port,
        perception_mode,
        resolved_fr or "",
        enable_exposure_control=enable_exposure_control,
        mixed_policy=mixed_policy if perception_mode == "mixed" else None,
    )
    app = ActivePerceptionLoop(
        camera_id=camera_id,
        debug=debug,
        enable_exposure_control=enable_exposure_control,
        enable_pan_tilt=enable_pan_tilt,
        pan_tilt_port=pan_tilt_port,
        perception_mode=perception_mode,
        face_registry_dir=(
            resolved_fr if perception_mode in ("face", "mixed") else face_registry_dir
        ),
        face_match_threshold=face_match_threshold,
        primary_hysteresis_frames=primary_hysteresis_frames,
        mixed_policy=mixed_policy,
        enable_gesture_actions=enable_gesture_actions,
        privacy_blur_faces=privacy_blur_faces,
        privacy_blur_kernel=privacy_blur_kernel,
        privacy_blur_pad=privacy_blur_pad,
        privacy_blur_passes=privacy_blur_passes,
        enable_voice=enable_voice,
        voice_mode=voice_mode,
        voice_lang=voice_lang,
        voice_model=voice_model,
        voice_record_seconds=voice_record_seconds,
        voice_llm=voice_llm,
        voice_llm_model=voice_llm_model,
        voice_clarify=voice_clarify,
        voice_save_wav=voice_save_wav,
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
