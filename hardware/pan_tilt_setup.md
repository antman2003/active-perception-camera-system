## Pan-tilt setup (placeholder)

Record your mechanical limits and calibration here:

- **Pan min/max (deg)**:
- **Tilt min/max (deg)**:
- **Home pose (deg)**:
- **Servo orientation**: (which direction is +pan / +tilt)
- **Notes**: cable strain relief, vibrations, backlash, etc.

Recommended first steps:
1. Start with conservative limits (e.g. pan 60–120, tilt 60–120).
2. Verify motion is smooth with no binding.
3. Only then expand range.
4. First flash `hardware/arduino/pan_tilt_serial/pan_tilt_serial.ino` and confirm the boot self-test moves the servos before testing Python serial control.
5. After the self-test, send `PING` or `P90 T90` over serial and expect `PONG` / `OK P90 T90`.


