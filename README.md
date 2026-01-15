# Active Perception Camera System

A camera-based **active perception system** that dynamically adjusts its sensing viewpoint based on perception uncertainty, closing the loop between sensing, decision-making, and physical action.

---

## Overview

Traditional computer vision systems operate on static, single-frame inputs.In contrast, **active perception** treats sensing as a decision-making process:  when perception confidence is low, the system actively changes *how* it senses the environment.

This project implements a **hardware-in-the-loop active perception pipeline** using a movable pan-tilt camera. The system continuously evaluates visual confidence and adapts its camera viewpoint to improve perception robustness under challenging real-world conditions such as:

- Low illumination  
- Motion blur  
- Limited depth of field  
- Suboptimal viewing angles  

The goal of this project is **system-level perception design**, not maximizing model accuracy.

---

## Key Concepts Demonstrated

- Active / embodied perception  
- Uncertainty-aware decision making  
- Closed-loop sensing–action systems  
- Camera viewpoint control as a perception strategy  
- Robust perception under real-world imaging noise  

---

## System Architecture

```
Camera → Perception → Uncertainty Estimation → Action Policy → Pan-Tilt Control
   ↑                                                               ↓
   └────────────────────────── Closed Perception Loop ─────────────┘
```

---

## Hardware Setup

### Core Hardware Components

| Component | Description |
|---------|-------------|
| Camera | Logitech Brio 100 USB Webcam |
| Pan-Tilt Platform | Yahboom 2-DOF Servo Pan-Tilt Kit |
| Microcontroller | Arduino Nano / Arduino Uno |
| Power Supply | External 5V supply (MB102 breadboard module) |
| Control Interface | USB Serial (PC ↔ Arduino) |

### Hardware Design Notes

- The camera is mounted on a 2-DOF pan-tilt platform to enable physical viewpoint changes.
- Servo motors are powered by an external 5V supply to ensure stability.
- Arduino handles low-level motor control; perception and policy logic run on the PC.
- Mounting prioritizes stability and modularity over mechanical precision.

---

## Software Architecture

### Software Stack

- **Language**: Python (PC), Arduino C++
- **Vision**: OpenCV
- **Communication**: pySerial
- **OS**: Windows 10

### Repository Structure

```
active-perception-camera-system/
├── README.md
├── hardware/
│   ├── wiring_diagram.md
│   └── pan_tilt_setup.md
├── src/
│   ├── camera.py
│   ├── perception.py
│   ├── uncertainty.py
│   ├── policy.py
│   ├── controller.py
│   └── main_loop.py
├── experiments/
│   ├── low_light_test.md
│   ├── blur_test.md
│   └── viewpoint_comparison.md
├── logs/
└── requirements.txt

```

---

## Active Perception Strategy

Instead of relying on lens autofocus or static inference, the system treats **viewpoint selection as a perception action**.

Typical loop:

1. Capture frame  
2. Run perception  
3. Estimate uncertainty  
4. Adjust viewpoint if confidence is low  
5. Re-evaluate perception  

This mirrors strategies used in robotics and embodied AI systems.

---

## Project Goals

- Demonstrate system-level perception thinking
- Explore sensing–action coupling
- Build a reusable foundation for embodied perception experiments

---

## Future Work

- Multi-view fusion
- Learned action policies
- Continuous viewpoint optimization
- Sensor fusion (IMU, depth)
- Mobile platforms

---

## Disclaimer

This project is for educational and experimental purposes only.

---


