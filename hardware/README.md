# Hardware Build Guide

This folder contains everything needed to build the physical side of the Active Perception Camera System: the bill of materials, the wiring, the Arduino firmware, and the calibration notes.

The Python stack only needs **a USB webcam** to run in digital-only mode. Adding the pan-tilt stage unlocks physical visual servoing and 2-DOF search, but is completely optional — `main.py` auto-detects the Arduino on startup and falls back to digital-only if none is found.

---

## Bill of Materials

| Component         | Recommended Part                              | Price (USD, Amazon 2026) |
| ----------------- | --------------------------------------------- | ------------------------ |
| Camera            | Logitech Brio 100 USB Webcam                  | ~$25                     |
| Pan-Tilt Platform | Yahboom 2-DOF Servo Pan-Tilt Kit (20/25 kg·cm)| ~$50                     |
| Microcontroller   | Arduino Nano / Arduino Uno                    | ~$20                     |
| Power Supply      | MB102 breadboard module (external 5V/6V)      | ~$10                     |
| Misc              | Jumper wires, small breadboard                | ~$5                      |

**Total**: ~$110.

> The Yahboom 20/25 kg·cm servos want roughly **6 V – 7.4 V** on the power rail. Do **not** run them off the Arduino 5V pin — it cannot supply enough current and you will get either no motion or juddering.

---

## Assembly Overview

```
+------------+       USB        +-----------+   PWM    +------------+
|  PC (Py)   |<---------------> |  Arduino  |--------->| Pan servo  |
|            |   115200 8N1     |  Nano/Uno |----+     +------------+
+------------+                  +-----------+    |
                                     |           +---> +------------+
                                     | GND / 5V        | Tilt servo |
                                     v                 +------------+
                                +-----------+
                                |  MB102    |  6V-7.4V rail to servos
                                |  5V PSU   |  GND common with Arduino
                                +-----------+
                                     ^
                                     | DC barrel jack
                                  (wall adapter)
```

Key rules:

- **Dedicated servo power.** Servos run off the external PSU, not the Arduino rail.
- **Common ground.** The external PSU ground and the Arduino ground must be tied together, otherwise signals are meaningless.
- **Signal pins.** `D9 → pan signal`, `D10 → tilt signal` (matches the firmware defaults).
- **Camera mounting.** The USB webcam sits on top of the tilt bracket; route the USB cable with enough slack to pan/tilt across the full range without binding.

See [`wiring_diagram.md`](wiring_diagram.md) for the wiring-level checklist and [`pan_tilt_setup.md`](pan_tilt_setup.md) for mechanical calibration.

---

## Firmware

The Arduino firmware lives in [`arduino/pan_tilt_serial/pan_tilt_serial.ino`](arduino/pan_tilt_serial/pan_tilt_serial.ino). It is intentionally minimal:

- On boot: attaches both servos, moves to home, runs a short self-test that walks pan and tilt to their limits.
- Accepts newline-terminated commands on USB serial at **115200 baud**:
  - `P<angle> T<angle>` — set both axes, e.g. `P90 T100`
  - `P<angle>` or `T<angle>` — update one axis only
  - `PING` — replies with `PONG` (liveness check)
- Replies:
  - `READY P90 T90` once on boot
  - `OK P90 T100` after each successful move
  - `ERR <line>` on unparsable input
- **Software limits (firmware side)** — clamped in `writePose()`:
  - Pan: `0° – 180°`
  - Tilt: `50° – 130°`
  - Home: `(90, 90)`

Update the `PAN_MIN / PAN_MAX / TILT_MIN / TILT_MAX` constants in the `.ino` file and re-flash if you need a different mechanical range. Remember to also update `pan_limits` / `tilt_limits` in `src/controller.py::HardwareConfig` so the Python side agrees.

### Flashing

1. Open `arduino/pan_tilt_serial/pan_tilt_serial.ino` in the Arduino IDE.
2. Select **Board: Arduino Nano** (or Uno), and the correct **Port** (e.g. `COM3`).
3. Upload. The servos should twitch through the self-test immediately after the upload finishes.

---

## Bringing It Up

1. **Power on** the external PSU first, **then** plug in the Arduino USB. This avoids back-feeding the servos through the USB rail at startup.
2. **Verify the self-test** visually — both axes should briefly walk to their software limits and return to home (90, 90).
3. **Liveness check** from Python:

   ```powershell
   python -c "from src.controller import HardwareController; c = HardwareController(port='COM3'); c.connect(); print(c.send_ping()); c.close()"
   ```

   Expected output: `PONG`.

4. **Interactive control** (useful for mechanical calibration):

   ```powershell
   python -c "from src.controller import run_interactive_stage_control; run_interactive_stage_control(port='COM3')"
   ```

   Commands inside the prompt: `90 90`, `p 45`, `t 110`, `c` (re-home), `q` (quit).

5. **Full system** (auto-detects the stage):

   ```powershell
   python main.py --port COM3
   ```

---

## Calibration Checklist

Fill in the real numbers for your build in [`pan_tilt_setup.md`](pan_tilt_setup.md):

- [ ] Pan min / max (deg) — soft limits you actually hit without binding.
- [ ] Tilt min / max (deg).
- [ ] Home pose that looks mechanically level (usually `90, 90`, but your mount may be off).
- [ ] Which direction is `+pan` (camera turning right vs left) and `+tilt` (looking up vs down). Flipping the sign of `pan_tilt_gain_pan` / `pan_tilt_gain_tilt` in `src/loop.py` handles mirrored servos without re-soldering.
- [ ] Any cable-strain or wobble notes.

Start with conservative limits (pan 60–120, tilt 60–120) for first motion tests, then expand after the mechanics look clean.

---

## Troubleshooting

### Servos don't move at all

- PSU not powered, or not shared ground with the Arduino. Check with a multimeter — you should measure ~6 V between the servo red rail and the Arduino GND.
- USB cable is data-only (charging cable). Swap for a data-capable one.
- Wrong `COM` port in Python. List ports with `python -c "from serial.tools import list_ports; [print(p.device) for p in list_ports.comports()]"`.

### Servos judder or reset the Arduino

- Almost always undervoltage — the servo drains the rail below USB's 5 V and the Arduino browns out. Use a PSU rated for at least 2 A at your servo voltage.

### Camera image looks frozen / black

- Another program is holding the camera. On Windows you can find the offender with `tools/handle/handle64.exe` (untracked, download from Sysinternals).
- Try `python main.py --cam 0` to fall back to your built-in webcam.

### Pan/tilt angles drift out of range on restart

See the project-level README’s **Troubleshooting** section — it's a software/firmware sync issue that is already patched in `HardwareController.connect()` / `close()`. The symptom is most often caused by the servos being mid-move when a previous session crashed.
