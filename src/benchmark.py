"""
Benchmark runner for comparing static and active perception variants.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.loop import ActivePerceptionLoop


BENCHMARK_VARIANTS: Dict[str, Dict[str, bool]] = {
    "static": {"enable_exposure_control": False, "enable_zoom_control": False},
    "active_exp": {"enable_exposure_control": True, "enable_zoom_control": False},
    "active_full": {"enable_exposure_control": True, "enable_zoom_control": True},
}


def run_benchmark(
    camera_id: int = 1,
    duration_s: float = 10.0,
    system: str = "all",
    debug: bool = False,
    label: str | None = None,
    distance_cm: float | None = None,
    lux: float | None = None,
) -> List[dict]:
    variants = list(BENCHMARK_VARIANTS.keys()) if system == "all" else [system]
    results: List[dict] = []

    print("=" * 60)
    print("Benchmark Mode")
    print("=" * 60)
    print(f"Camera: {camera_id}")
    print(f"Duration per run: {duration_s:.1f}s")
    print(f"Variants: {', '.join(variants)}")
    print("Keep marker position and scene conditions as consistent as possible.")
    print("=" * 60)

    for idx, variant in enumerate(variants):
        if idx > 0:
            answer = input(f"\nPress Enter to start '{variant}' (or type q to stop): ").strip().lower()
            if answer == "q":
                break

        flags = BENCHMARK_VARIANTS[variant]
        print(f"\n[Benchmark] Running variant: {variant}")
        app = ActivePerceptionLoop(
            camera_id=camera_id,
            debug=debug,
            enable_exposure_control=flags["enable_exposure_control"],
            enable_zoom_control=flags["enable_zoom_control"],
            show_window=True,
        )
        summary = app.run(duration_s=duration_s)
        summary.update(
            {
                "variant": variant,
                "camera_id": camera_id,
                "label": label,
                "distance_cm": distance_cm,
                "lux": lux,
            }
        )
        results.append(summary)
        _print_summary(summary)

    if results:
        output_path = _save_results(results, label=label)
        print(f"\n[Benchmark] Results saved to: {output_path}")

    return results


def _print_summary(summary: dict) -> None:
    time_to_stable = summary["time_to_stable_s"]
    best_stable_u = summary["best_stable_uncertainty"]
    final_stable_u = summary["final_stable_uncertainty"]
    stable_samples = summary["stable_samples"]

    print(
        "  best_stable_u={}, final_stable_u={}, time_to_stable={}, "
        "search_ratio={:.3f}, stable_samples={}, final_zoom={:.1f}x".format(
            f"{best_stable_u:.3f}" if best_stable_u is not None else "n/a",
            f"{final_stable_u:.3f}" if final_stable_u is not None else "n/a",
            f"{time_to_stable:.2f}s" if time_to_stable is not None else "n/a",
            summary["search_frames_ratio"],
            stable_samples,
            summary["final_zoom"],
        )
    )
    print(
        "  aux_detected_rate={:.3f}, aux_confirmed_detected_rate={:.3f}, "
        "avg_raw_u={:.3f}, avg_size={:.1f}".format(
            summary["detected_rate"],
            summary["confirmed_detected_rate"],
            summary["avg_raw_uncertainty"],
            summary["avg_size_when_detected"],
        )
    )


def _save_results(results: List[dict], label: str | None = None) -> str:
    root = Path("logs") / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    path = root / f"{ts}{suffix}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    return str(path)
