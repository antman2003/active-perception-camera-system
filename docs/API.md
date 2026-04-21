# API Reference

This document describes the **public** surface of the project — the things you would `import` in your own code, or pass on the command line. Internal helpers (`_send_pose`, `_drain_input`, etc.) are omitted on purpose; if a method is listed here, it is safe to rely on.

- [Command-line interface](#command-line-interface)
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
               [--perception {aruco,face}] [--face-registry DIR] [--face-threshold FLOAT]
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
| `--perception`   | `aruco` | full          | `aruco` = markers; `face` = registry + LBPH (needs `--face-registry`).   |
| `--face-registry` | —     | full          | Root folder: one subfolder per person with face images (see `face_registry/README.txt`). |
| `--face-threshold` | `85.0` | full       | LBPH **distance** cutoff; **lower distance = better match**; above → HUD shows `?`. |

Press **`q`** in the video window to exit. The stage auto-homes on shutdown.

### `demo.py`

```
python demo.py [--cam INT] [--debug] [--port STR] [--no-pan-tilt]
               [--perception {aruco,face}] [--face-registry DIR] [--face-threshold FLOAT]
```

Same camera / debug / pan-tilt semantics as `main.py --mode full`.

---

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
| `perception_mode`         | `str`  | `"aruco"` | `"aruco"` or `"face"`.                                                                      |
| `face_registry_dir`       | `str \| None` | `None` | Required when `perception_mode=="face"`.                                                    |
| `face_match_threshold`    | `float` | `85.0` | LBPH distance threshold (lower is better).                                                |

If `enable_pan_tilt=True` but the port cannot be opened, the loop prints a warning and continues in digital-only mode — it never raises.

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
| `create_perception(mode, face_registry_dir=..., face_match_threshold=...)` | Factory used by `ActivePerceptionLoop`. |

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
