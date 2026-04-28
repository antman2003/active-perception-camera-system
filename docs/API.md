# API Reference

This document describes the **public** surface of the project — the things you would `import` in your own code, or pass on the command line. Internal helpers (`_send_pose`, `_drain_input`, etc.) are omitted on purpose; if a method is listed here, it is safe to rely on.

- [Command-line interface](#command-line-interface)
- [Voice — CLI flags (Session 30)](#voice--cli-flags-session-30)
- [Voice / ASR (`docs/VOICE_ASR.md`)](VOICE_ASR.md)
- [Voice intent — text → JSON (`docs/VOICE_INTENT.md`)](VOICE_INTENT.md)
- [Voice executor — JSON → hardware](#voice-executor--json--hardware)
- [`ActivePerceptionLoop`](#activeperceptionloop)
- [Perception backends (`src/perception/`)](#perception-backends-srcperception)
- [`HardwareController`](#hardwarecontroller)
- [Arduino serial protocol](#arduino-serial-protocol)

---

## Command-line interface

Two entry scripts live at the repo root:

- **`main.py`** — unified CLI (full system, uncertainty inspection, policy demo, benchmark).
- **`demo.py`** — thin wrapper that only runs the full loop.

Both auto-detect a pan-tilt stage by briefly opening the serial port on startup. If it opens, the stage is enabled; otherwise the run falls back to pure digital mode.

### `main.py`

```
python main.py [--mode {full,uncertainty,policy,benchmark}]
               [--cam INT] [--debug]
               [--port STR] [--no-pan-tilt]
               [--perception {aruco,face,mixed,auto}] [--face-registry DIR] [--face-threshold FLOAT]
               [--mixed-policy {aruco_first,face_first,larger_area}]
               [--no-auto-exposure] [--face-primary-hysteresis N]
               [--system {all,static,active_exp,active_full}]
               [--duration FLOAT] [--label STR]
               [--distance-cm FLOAT] [--lux FLOAT]
```

| Flag             | Default | Used by       | Meaning                                                                 |
| ---------------- | ------- | ------------- | ----------------------------------------------------------------------- |
| `--mode`         | `full`  | all           | Which subsystem to run.                                                 |
| `--cam`          | `1`     | all           | OpenCV camera index. `0` usually = built-in laptop cam.                 |
| `--debug`        | off     | full, benchmark | Per-frame blackbox logging into `logs/blackbox/<timestamp>/`.          |
| `--port`         | `COM3`  | full          | Serial port to probe for the pan-tilt Arduino.                          |
| `--no-pan-tilt`  | off     | full          | Force pure digital mode even if an Arduino is present.                  |
| `--system`       | `all`   | benchmark     | Which system variant to benchmark.                                      |
| `--duration`     | `10.0`  | benchmark     | Seconds per benchmark run.                                              |
| `--label`        | —       | benchmark     | Free-form tag saved with the result.                                    |
| `--distance-cm`  | —       | benchmark     | Manual distance annotation.                                             |
| `--lux`          | —       | benchmark     | Manual illuminance annotation.                                          |
| `--perception`   | `mixed` | full          | Default **`mixed`**: ArUco + face each frame, one **active** target (see `--mixed-policy`). Also `aruco`, `face`, `auto` (=mixed). |
| `--mixed-policy` | `aruco_first` | full   | With `mixed`/`auto`: `aruco_first`, `face_first`, or `larger_area` (compare primary contour areas). |
| `--face-registry` | (default) | full   | Root folder: one subfolder per person with face images (see `face_registry/README.txt`). **Omitted** in `face`/`mixed`/`auto` → `<repo>/face_registry` (must exist). |
| `--face-threshold` | `85.0` | full       | LBPH **distance** cutoff; **lower distance = better match**; above → HUD shows `?`. |
| `--no-auto-exposure` | off | full      | Skips `ExploreExposureState` (fixed exposure index from startup). Useful to compare LBPH stability against auto face-tuned sweeps. |
| `--face-primary-hysteresis` | `0` | full | Face mode only: when **two** faces are similar size (≥88% area ratio) and different identities, require **N** consecutive frames before switching the locked primary. `0` disables. |

Press **`q`** in the video window to exit. The stage auto-homes on shutdown.

### `demo.py`

```
python demo.py [--cam INT] [--debug] [--port STR] [--no-pan-tilt]
               [--perception {aruco,face,mixed,auto}] [--face-registry DIR] [--face-threshold FLOAT]
               [--mixed-policy {aruco_first,face_first,larger_area}]
               [--no-auto-exposure] [--face-primary-hysteresis N]
```

Same camera / debug / pan-tilt semantics as `main.py --mode full`.

---

## Voice — CLI flags (Session 30)

`main.py` and `demo.py` accept the same voice-related options (PTT is the **`v`** key in the OpenCV window):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--voice` | off | Start the background voice worker (recording → ASR → intent → executor). |
| `--voice-mode` | `ptt` | `ptt`: press `v` to record. `always`: wake word + VAD capture (Session 31). |
| `--voice-lang` | `zh` | ASR hint: `zh`, `en`, or `auto` (Whisper auto-detect). |
| `--voice-model` | `base` | `faster-whisper` model size (`tiny` … `large-v3`; see `docs/VOICE_ASR.md`). |
| `--voice-record-seconds` | `5.0` | Fixed recording window per PTT press. |
| `--voice-save-wav` | off | Write each PTT capture under `logs/blackbox/<session>/audio/` (debug). |
| `--voice-llm` / `--no-voice-llm` | on | Allow **local Ollama** fallback when rules-only parsing yields no actionable command (see `docs/VOICE_INTENT.md`). |
| `--voice-llm-model` | `qwen2.5:1.5b` | Ollama model tag. |
| `--voice-clarify` / `--no-voice-clarify` | off | One-round clarification: first unclear transcript → HUD prompt only (no LLM, no servo); second PTT may merge into one LLM call if enabled. |

**ASR in the live loop:** the PTT worker always constructs `FasterWhisperAsrProvider`. `MockAsrProvider` is for **`python -m src.voice.asr_demo --backend mock`** and unit tests (`tests/test_voice_asr_provider.py`), not wired to `main.py` by default.

### Priority vs vision PT, gestures, and privacy

| Writer | When it runs | Interaction |
| --- | --- | --- |
| **Voice executor** | After validated JSON, on PTT completion | Sets `voice_pt_suppress_until` on the loop; **visual face-tracking PT** (`pan_tilt.nudge` in `MonitorState`) **skips writes** while `time.time() < voice_pt_suppress_until`. |
| **Gesture actions** | When enabled (`--gesture-actions`, default on) | Same suppression flag applies where implemented in `states.py` (do not fight voice moves). |
| **`--privacy` / blur** | Display / zoom policy only | Does **not** disable voice; ASR still runs **locally**. Blur affects on-screen face ROIs, not the microphone path. |

### Blackbox events (voice)

With **`--debug`**, per-frame `frames.jsonl` is heavy; **voice events always append to `events.jsonl`** whenever blackbox is enabled (normal runs create a session folder).

| `event_type` | Role |
| --- | --- |
| `voice_ptt_triggered` | User pressed `v`; includes `awaiting_clarification`, model, LLM/clarify flags. |
| `voice_audio_recorded` | RMS / peak / optional `wav_path`. |
| `voice_asr_done` | `t_asr_ms`, `language`, `duration_audio_s`, transcript. |
| `voice_text_normalized` | Chinese offline normalization applied (`before` / `after`). |
| `voice_llm_plan` | Whether LLM is **planned** on this turn (first-turn heuristic). |
| `voice_llm_*_attempted` / `voice_llm_*_result` | Bundle / text-fix / legacy `propose_command` timing and raw payload (wrapper in `worker.py`). |
| `voice_clarify_prompt` | Clarification-only turn (no execution). |
| `voice_intent_resolved` | Final `parser` (`rule` / `llm` / `none`) + `commands`. |
| `voice_intent_schema_reject` | LLM was used but output did not validate (`note == llm_failed`). |
| `voice_command_*` | `execute_voice_command`: `voice_command_execute`, `voice_command_noop`, `voice_command_search`, `voice_command_skipped_no_pantilt`, `voice_command_rejected`, `voice_command_failed`. |
| `voice_execute_done` | Batch finished (`n_commands`). |
| `voice_ptt_cycle_end` | **Cumulative counters** snapshot for this PTT (`outcome`, `metrics` dict). |
| `voice_metrics_session_final` | Same counters at **session shutdown** (if `--voice` was on). |

Counter keys are sparse integers (e.g. `voice_asr_done`, `voice_intent_parser_rule`, `voice_llm_bundle_attempts`, `voice_exec_hardware`, `voice_clarify_prompts`). See `src/voice/metrics.py` and `src/voice/worker.py` for the authoritative list.

---

## Voice executor — JSON → hardware

Module: `src/voice/executor.py`

Consumes **validated** Command JSON (from `src/voice_intent/schema.py`) and:

- Calls `HardwareController` (pan-tilt) when present.
- Or sets `voice_request_search` on the loop context to trigger an FSM transition.
- If `pan_tilt is None`, it must **not crash** and must not touch serial.

Public functions:

- `execute_voice_command(cmd, context, *, pt_suppress_s=0.6, smooth=True) -> VoiceExecutionReport`
- `execute_voice_commands(commands, context, *, pt_suppress_s=0.6, smooth=True) -> list[VoiceExecutionReport]`

## `ActivePerceptionLoop`

Module: `src/loop.py`

The top-level orchestrator. Wires the camera, perception, uncertainty, policy, state machine, (optional) hardware controller, and blackbox logger together.

```python
from src.loop import ActivePerceptionLoop

app = ActivePerceptionLoop(
    camera_id=1,
    debug=False,
    enable_exposure_control=True,
    enable_zoom_control=True,
    enable_pan_tilt=False,
    pan_tilt_port="COM3",
    show_window=True,
)
app.run()
```

### Constructor parameters

| Parameter                 | Type   | Default  | Effect                                                                                     |
| ------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------ |
| `camera_id`               | `int`  | `1`      | OpenCV camera index.                                                                       |
| `debug`                   | `bool` | `False`  | Enables per-frame blackbox logging (heavier I/O).                                          |
| `enable_exposure_control` | `bool` | `True`   | If `False`, the FSM never enters `ExploreExposureState`.                                   |
| `enable_zoom_control`     | `bool` | `True`   | If `False`, zoom + sniper branches are disabled.                                           |
| `enable_pan_tilt`         | `bool` | `False`  | If `True`, tries to open the serial port and enables physical servoing + physical search. |
| `pan_tilt_port`           | `str`  | `"COM3"` | Serial port name for the Arduino.                                                          |
| `show_window`             | `bool` | `True`   | If `False`, runs headless (no `cv2.imshow`).                                               |
| `perception_mode`         | `str`  | `"aruco"` | Constructor default remains `"aruco"`; CLI `main.py` / `demo.py` default **`mixed`**. Values: `aruco`, `face`, `mixed`, `auto`. |
| `mixed_policy`            | `str`  | `"aruco_first"` | Only `mixed`/`auto`: priority for the active tracking target. |
| `face_registry_dir`       | `str \| None` | `None` | For `face` / `mixed` / `auto`, `None`/empty → `<repo>/face_registry` (see `src/face_registry_resolve.py`). |
| `face_match_threshold`    | `float` | `85.0` | LBPH distance threshold (lower is better).                                                |
| `primary_hysteresis_frames` | `int` | `0`   | Passed to `FaceDetector` in face mode (see CLI flag). Ignored for ArUco.                 |
| `enable_voice` | `bool` | `False` | Start the voice worker and allocate `voice_obs` counters. |
| `voice_mode` | `str` | `\"ptt\"` | `ptt` (Session 30) or `always` (Session 31 wake word + VAD). |
| `voice_lang` | `str` | `"zh"` | ASR language hint (`auto` → `None` for Whisper). |
| `voice_model` | `str` | `"base"` | Whisper model size. |
| `voice_record_seconds` | `float` | `5.0` | PTT window. |
| `voice_llm` | `bool` | `True` | Enable Ollama fallback path. |
| `voice_llm_model` | `str` | `"qwen2.5:1.5b"` | Ollama model name. |
| `voice_clarify` | `bool` | `True` | One-round clarification (see voice CLI above). |
| `voice_save_wav` | `bool` | `False` | Save PTT WAV under the blackbox session folder. |

If `enable_pan_tilt=True` but the port cannot be opened, the loop prints a warning and continues in digital-only mode — it never raises.

### Face vs ArUco exposure (Session 26)

ArUco exposure sweep minimizes **geometry uncertainty** (`UncertaintyEngine` on marker corners). Face mode uses the same sweep machinery but **scores each exposure** with extra terms so winners stay closer to **enrollment-friendly** lighting:

- **`face_exposure_lbph_weight`** (default `0.12`): adds a term proportional to **LBPH distance** on the primary face (when detected), so lower distance beats marginally lower `raw_u`.
- **`face_exposure_brightness_weight`** (default `0.10`): penalizes **very bright** primary-face ROIs (mean V in HSV) to reduce skin blow-out that hurts Haar/LBPH vs registry crops.
- **`exposure_tiebreak_preferred_val`**: among near-tie winners, prefer the hardware exposure level closest to this OpenCV log value (face default **-6.0**, ArUco **-4.0** — shorter exposure bias for skin).
- **`face_exposure_indices`**: optional `list[int]` of indices into `policy.exposure_levels`; when set in **face** mode only, the sweep visits that subset (faster calibration).

**Recommended tuning order (face):** (1) enroll under similar light to the demo room; (2) run with auto exposure on and trigger 1–2 sweeps (change room light or temporarily cover lens); (3) if names flip to `?` after sweep, lower `face_match_threshold` slightly or tighten `face_exposure_brightness_weight` / `exposure_tiebreak_preferred_val`; (4) use `--no-auto-exposure` to confirm the regression is exposure-related.

**UncertaintyEngine** uses `FACE_UNCERTAINTY_PARAMS` / `ARUCO_UNCERTAINTY_PARAMS` in `src/uncertainty.py` (avoid duplicating magic numbers in `loop.py`). In **`mixed`/`auto`** mode, the engine switches with `perception.active_backend` (`"aruco"` vs `"face"`).

**Mixed / auto:** both targets are drawn; **zoom / pan-tilt / exposure** follow the **active** target only. This is not “fuse two distances into one LBPH score”—it is **one primary ROI per frame** chosen by `mixed_policy`.

**Pan-tilt (face defaults on the loop):** `pan_tilt_gain_pan` **6.5**, `pan_tilt_gain_tilt` **4.5**, `pan_tilt_deadzone` **0.055** (ArUco defaults remain **8.0 / 5.0 / 0.05** if you start in marker mode). **Physical search** uses a **narrower tilt grid** in face mode (desk / standing height band).

### Tuning knobs (public attributes)

Set these after construction, before `run()`, to tweak behavior without editing the FSM code:

| Attribute                          | Default | Controls                                                        |
| ---------------------------------- | ------- | --------------------------------------------------------------- |
| `monitor_explore_enter_threshold`  | `0.60`  | Uncertainty needed to trigger exposure sweep.                   |
| `monitor_explore_exit_threshold`   | `0.50`  | Hysteresis lower bound for exposure sweep.                      |
| `monitor_zoom_enter_threshold`     | `0.55`  | Uncertainty that forces a zoom sweep at base zoom.              |
| `monitor_zoom_exit_threshold`      | `0.45`  | Hysteresis lower bound for zoom sweep.                          |
| `monitor_roi_lost_threshold`       | `8`     | Frames of ROI loss at zoom > 1x before triggering Sniper.       |
| `monitor_nudge_gain`               | `0.15`  | Smoothness of digital ROI visual servoing.                      |
| `pan_tilt_gain_pan`                | `8.0`   | P-gain from pixel error to pan degrees. Flip sign to mirror.    |
| `pan_tilt_gain_tilt`               | `5.0`   | P-gain from pixel error to tilt degrees. Flip sign to mirror.   |
| `pan_tilt_deadzone`                | `0.05`  | Normalized pixel deadzone (5% of frame) to kill micro-jitter.   |
| `pan_tilt_search_lost_threshold`   | `30`    | Frames of no detection at zoom ≤ 1x before Physical Search.     |
| `sniper_timeout_frames`            | `60`    | Frames Sniper waits before escalating to Physical Search.       |
| `exposure_settle_frames`           | `2`     | Frames to wait after changing exposure before sampling.         |
| `exposure_sample_frames`           | `3`     | Frames averaged per exposure candidate.                         |
| `zoom_settle_frames`               | `1`     | Same, for zoom sweep.                                           |
| `zoom_sample_frames`               | `2`     | Same, for zoom sweep.                                           |
| `detect_confirm_frames`            | `2`     | Consecutive detections before `confirmed_detected = True`.      |
| `lost_confirm_frames`              | `3`     | Consecutive misses before `confirmed_lost = True`.              |
| `exposure_tiebreak_preferred_val`  | `-6` face / `-4` aruco | Near-tie exposure winner prefers level closest to this value. |
| `face_exposure_lbph_weight`        | `0.12`  | Face sweep only: weight on normalized LBPH distance.            |
| `face_exposure_brightness_weight`  | `0.10`  | Face sweep only: weight on ROI highlight penalty.               |
| `face_exposure_indices`            | `None`  | Face only: optional list of exposure indices to sweep.         |

### Methods

- `run()` — blocks until the user presses `q` or the camera closes. Handles graceful shutdown: homes the pan-tilt (if any), closes the serial port, releases the camera.

---

## Perception backends (`src/perception/`)

Session 25 layout: **one `detect` / `visualize` contract** for the loop; ArUco and face both emit ArUco-shaped `corners` for `UncertaintyEngine` / states.

| Symbol | Role |
| ------ | ---- |
| `PerceptionDetector` | ABC: `detect`, `visualize`. |
| `ArucoDetector` | Marker detection (default). Alias export: `PerceptionSystem`. |
| `FaceDetector` | Haar frontal face + LBPH; largest face = primary target; HUD shows name or `?`. |
| `CombinedPerception` | Runs ArUco + face; sets `active_backend`; `detect` returns the **chosen** primary only. |
| `create_perception(..., mixed_policy=...)` | Factory; `mode` includes `mixed` / `auto`. |

Smoke test (ArUco only): `python -m src.perception` (opens camera index `1`).

**Dependency:** face mode needs `opencv-contrib-python` (provides `cv2.face`).

---

## `HardwareController`

Module: `src/controller.py`

Hardware abstraction layer for the pan-tilt stage. Handles serial I/O, software limits, smoothing, and context management. **Use this class**, not raw `pyserial`, if you are writing any new hardware code.

### Dataclasses

```python
@dataclass(frozen=True)
class PanTiltPose:
    pan: int      # degrees
    tilt: int     # degrees

@dataclass(frozen=True)
class MotionProfile:
    step_deg: int = 10         # max degrees per smooth step
    step_settle_s: float = 0.25  # pause between smooth steps
    settle_s: float = 0.3        # pause after a non-smooth move

@dataclass(frozen=True)
class HardwareConfig:
    port: str = "COM3"
    baudrate: int = 115200
    pan_limits:  Tuple[int, int] = (0, 180)
    tilt_limits: Tuple[int, int] = (50, 130)
    home_pose:   PanTiltPose    = PanTiltPose(90, 90)
    startup_delay_s: float = 2.0
    timeout_s: float = 1.0
    motion: MotionProfile = MotionProfile()
```

### Construction and lifecycle

```python
from src.controller import HardwareController, MotionProfile

# Canonical usage: context manager auto-connects and auto-homes + closes.
with HardwareController(port="COM3") as ctl:
    ctl.move_to(120, 100, smooth=True)
    ctl.home()
```

- `__init__(port, baudrate, pan_limits, tilt_limits, home_pose, startup_delay_s, timeout_s, motion_profile)` — all keyword args; defaults match the firmware.
- `connect()` — open the serial port, wait for the Arduino boot delay, drain the boot banner, **and send a home command so software and hardware state are aligned**.
- `close()` — sends a final home command, then closes the port.
- `__enter__` / `__exit__` — same as `connect()` / `close()`.

### Properties

- `is_connected: bool` — `True` if the serial port is open.
- `current_pose: PanTiltPose` — the last pose the controller successfully sent (its best guess of where the stage is).

### Movement methods

All movement methods clamp targets through `clamp_pose()` before sending, so you cannot drive the stage past the configured software limits.

| Method                                                                    | What it does                                                                                                                                          |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `move_to(pan=None, tilt=None, settle_s=None, smooth=False, step_deg=None, step_settle_s=None)` | Absolute move. `None` axis is left alone. With `smooth=True` it breaks the motion into `step_deg` chunks with a short settle between each. |
| `move_by(delta_pan=0, delta_tilt=0, smooth=False, settle_s=None, step_deg=None, step_settle_s=None)` | Relative move. Internally does `move_to(current + delta)`.                                                                                      |
| `nudge(delta_pan=0, delta_tilt=0, settle_s=0.15)`                         | Short, fast relative move. Used by visual servoing every frame.                                                                                       |
| `home(smooth=True, settle_s=None)`                                        | Go back to the configured home pose. Smooth by default because home is usually the farthest point from wherever you were.                             |
| `center(settle_s=0.3)`                                                    | Alias of `home()` kept for backwards compatibility.                                                                                                   |
| `set_pan_tilt(pan=None, tilt=None, settle_s=0.2)`                         | Alias of `move_to(...)` (legacy name).                                                                                                                |
| `move_smoothly(pan, tilt, step_deg=10, step_settle_s=0.1)`                | Low-level stepped move. Prefer `move_to(..., smooth=True)` in new code.                                                                               |
| `sweep_poses(poses, smooth=True, step_deg=None, step_settle_s=None, settle_s=None)` | Runs an iterable of `PanTiltPose` in order, returns the list of Arduino replies.                                                            |

### Support methods

- `clamp_pose(pan=None, tilt=None) -> PanTiltPose` — apply software limits without moving.
- `send_ping() -> str` — sends `PING`, returns the reply (`PONG` when healthy). Useful for liveness checks.

### Convenience demos

- `run_servo_demo(port="COM3", ...)` — homes, sweeps pan to both limits, then tilt to both limits. Good first smoke test after flashing firmware.
- `run_interactive_stage_control(port="COM3", ...)` — terminal REPL for manual control. Commands: `<pan> <tilt>`, `p <pan>`, `t <tilt>`, `c`, `q`.

---

## Arduino serial protocol

The firmware in `hardware/arduino/pan_tilt_serial/pan_tilt_serial.ino` speaks a tiny ASCII protocol over USB serial at **115200 8N1**, newline-terminated.

### Commands (PC → Arduino)

| Command         | Example     | Effect                                                                 |
| --------------- | ----------- | ---------------------------------------------------------------------- |
| `P<int> T<int>` | `P90 T100`  | Set both axes (degrees).                                               |
| `P<int>`        | `P45`       | Set pan only.                                                          |
| `T<int>`        | `T110`      | Set tilt only.                                                         |
| `PING`          | `PING`      | Liveness check.                                                        |

All angles are clamped server-side to the firmware limits (`PAN_MIN/MAX`, `TILT_MIN/MAX`).

### Replies (Arduino → PC)

| Reply                   | When                                                   |
| ----------------------- | ------------------------------------------------------ |
| `READY P90 T90`         | Once on boot, after the self-test finishes.            |
| `OK P<pan> T<tilt>`     | After a successful move. Values are the post-clamp pose. |
| `PONG`                  | Reply to `PING`.                                       |
| `ERR <line>`            | The command could not be parsed.                       |

### Behavior on boot

1. Attach both servos.
2. Move to home (`90, 90`).
3. Run `runSelfTest()`: walk pan to its limits and back, walk tilt to its limits and back.
4. Print `READY P90 T90`.
5. Enter the main loop, waiting for `Serial.readStringUntil('\n')`.

Because the Arduino also auto-resets on USB open (DTR line), `HardwareController.connect()` inserts a `startup_delay_s` (default **2 s**) before issuing its first command.
