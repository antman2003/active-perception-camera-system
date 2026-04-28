"""
CLI: transcribe a WAV file, record from the default mic, or run mock ASR.

  python -m src.voice.asr_demo --backend mock
  python -m src.voice.asr_demo --backend faster --input path/to.wav --lang zh
  python -m src.voice.asr_demo --backend faster --record-seconds 3 --lang zh

**Default real ASR model:** faster-whisper with Whisper **``base``** (override with ``--model``).
**Where weights live:** Hugging Face Hub cache (not inside this repo); see ``docs/VOICE_ASR.md``.
**How to verify step-by-step:** same file ``docs/VOICE_ASR.md`` §3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.voice.asr_provider import (
    AsrProvider,
    FasterWhisperAsrProvider,
    MockAsrProvider,
)


# Formats we decode by path (ffmpeg / faster-whisper); not libsndfile-only WAV.
_INPUT_VIA_PATH_SUFFIXES = frozenset(
    {".m4a", ".mp3", ".mp4", ".aac", ".opus", ".webm", ".flac", ".ogg", ".mkv"}
)


def _load_wav_16k_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data), int(sr)


def _record_seconds(seconds: float, samplerate: int = 16000) -> np.ndarray:
    import sounddevice as sd

    frames = int(seconds * samplerate)
    audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def _build_provider(args: argparse.Namespace) -> AsrProvider:
    if args.backend == "mock":
        return MockAsrProvider(text=args.mock_text)
    return FasterWhisperAsrProvider(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=not args.no_vad,
    )


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Session 30: ASR demo (mock or faster-whisper)")
    p.add_argument(
        "--backend",
        choices=("mock", "faster"),
        default="faster",
        help="mock: no model load; faster: faster-whisper (default: faster)",
    )
    p.add_argument("--mock-text", default="回家", help="Transcript returned by mock backend")
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to audio file. .wav is read via soundfile; .m4a/.mp3/... are decoded by faster-whisper (ffmpeg).",
    )
    p.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Record from default microphone for SEC seconds (16 kHz mono)",
    )
    p.add_argument("--model", default="base", help="faster-whisper model size (tiny/base/small/...)")
    p.add_argument("--device", default="cpu", help="faster-whisper device, e.g. cpu or cuda")
    p.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute_type (cpu: int8/float32; cuda: float16)",
    )
    p.add_argument(
        "--lang",
        default="auto",
        help='Language hint: zh, en, or "auto" for detection',
    )
    p.add_argument("--no-vad", action="store_true", help="Disable faster-whisper VAD filter")
    args = p.parse_args(argv)

    if args.input and args.record_seconds is not None:
        print("Use only one of --input or --record-seconds", file=sys.stderr)
        return 2

    if args.input is None and args.record_seconds is None:
        if args.backend == "mock":
            args.record_seconds = 0.0
        else:
            p.print_help()
            print(
                "\nError: provide --input PATH or --record-seconds SEC (mock may omit both).",
                file=sys.stderr,
            )
            return 2

    prov = _build_provider(args)
    lang = None if args.lang.lower() == "auto" else args.lang

    try:
        if args.input:
            path = Path(args.input).expanduser()
            if not path.is_file():
                print(f"File not found: {path}", file=sys.stderr)
                return 1
            suffix = path.suffix.lower()
            if args.backend == "mock" or suffix in _INPUT_VIA_PATH_SUFFIXES:
                # faster-whisper decodes via ffmpeg; m4a/mp3/... are not read by soundfile here.
                result = prov.transcribe(str(path.resolve()), lang_hint=lang)
            else:
                pcm, sr = _load_wav_16k_mono(path)
                result = prov.transcribe(pcm, lang_hint=lang, sample_rate_hz=sr)
        else:
            assert args.record_seconds is not None
            if args.record_seconds > 0:
                pcm = _record_seconds(float(args.record_seconds))
                sr = 16000
            else:
                pcm = np.zeros(1, dtype=np.float32)
                sr = 16000
            result = prov.transcribe(pcm, lang_hint=lang, sample_rate_hz=sr)
    except Exception as e:
        print(f"ASR failed: {e}", file=sys.stderr)
        return 1

    print("--- ASR ---")
    print(f"text: {result.text!r}")
    print(f"t_asr_ms: {result.t_asr_ms}")
    print(f"language: {result.language!r}")
    print(f"duration_audio_s: {result.duration_audio_s!r}")
    print(f"n_segments: {len(result.segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
