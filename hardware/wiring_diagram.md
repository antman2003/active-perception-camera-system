## Wiring diagram (placeholder)

Minimum safe wiring checklist:
- **Servo power**: use an external regulated supply and do not power servos from Arduino 5V pin.
- **Yahboom note**: if you are using the Yahboom 20KG/25KG servos from the 2DOF PTZ kit, their docs recommend about **6V-7.4V** servo power. Undervoltage can cause no motion or unstable motion.
- **Common ground**: connect PSU GND ↔ Arduino GND ↔ servo GND.
- **Signal**: Arduino PWM pins to servo signal wires.
- **Suggested pins for this project**: `D9 -> pan`, `D10 -> tilt`.

Fill in:
- Servo model(s)
- Arduino board + pins used for pan/tilt
- Power supply rating and voltage


