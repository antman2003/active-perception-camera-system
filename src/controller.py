"""
Serial protocol + safe servo command API for pan/tilt servos.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import serial


class HardwareController:
    """
    Thin wrapper over the Arduino serial protocol.

    Expected command format:
        P90 T90
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        pan_limits: Tuple[int, int] = (60, 120),
        tilt_limits: Tuple[int, int] = (60, 120),
        startup_delay_s: float = 2.0,
        timeout_s: float = 1.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.startup_delay_s = startup_delay_s
        self.timeout_s = timeout_s

        self.serial_port: Optional[serial.Serial] = None
        self.current_pan = 90
        self.current_tilt = 90

    def connect(self) -> None:
        """Open the serial port and give the board time to reboot."""
        if self.serial_port is not None and self.serial_port.is_open:
            return

        self.serial_port = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        time.sleep(self.startup_delay_s)
        self._drain_input()

    def close(self) -> None:
        if self.serial_port is not None and self.serial_port.is_open:
            self.serial_port.close()
        self.serial_port = None

    def _require_connection(self) -> serial.Serial:
        if self.serial_port is None or not self.serial_port.is_open:
            raise RuntimeError("HardwareController is not connected.")
        return self.serial_port

    def _drain_input(self) -> str:
        port = self._require_connection()
        data = port.read_all()
        return data.decode(errors="ignore").strip()

    @staticmethod
    def _clamp(value: int, limits: Tuple[int, int]) -> int:
        lo, hi = limits
        return max(lo, min(hi, int(value)))

    def set_pan_tilt(
        self,
        pan: Optional[int] = None,
        tilt: Optional[int] = None,
        settle_s: float = 0.2,
    ) -> str:
        """
        Send one pan/tilt command.

        Returns any serial response captured shortly after the command.
        """
        port = self._require_connection()
        target_pan = self.current_pan if pan is None else self._clamp(pan, self.pan_limits)
        target_tilt = self.current_tilt if tilt is None else self._clamp(tilt, self.tilt_limits)

        cmd = f"P{target_pan} T{target_tilt}\n"
        port.write(cmd.encode("utf-8"))
        port.flush()
        time.sleep(settle_s)

        self.current_pan = target_pan
        self.current_tilt = target_tilt
        return self._drain_input()

    def center(self, settle_s: float = 0.3) -> str:
        return self.set_pan_tilt(90, 90, settle_s=settle_s)

    def nudge(self, delta_pan: float = 0.0, delta_tilt: float = 0.0, settle_s: float = 0.15) -> str:
        next_pan = round(self.current_pan + delta_pan)
        next_tilt = round(self.current_tilt + delta_tilt)
        return self.set_pan_tilt(next_pan, next_tilt, settle_s=settle_s)


def run_servo_demo(
    port: str = "COM3",
    baudrate: int = 115200,
    pan_limits: Tuple[int, int] = (60, 120),
    tilt_limits: Tuple[int, int] = (60, 120),
) -> None:
    """
    Send a conservative motion sequence: center -> small pan -> small tilt -> center.
    """
    controller = HardwareController(
        port=port,
        baudrate=baudrate,
        pan_limits=pan_limits,
        tilt_limits=tilt_limits,
    )
    controller.connect()
    try:
        sequence = [
            (90, 90),
            (80, 90),
            (100, 90),
            (90, 90),
            (90, 80),
            (90, 100),
            (90, 90),
        ]
        for pan, tilt in sequence:
            reply = controller.set_pan_tilt(pan, tilt, settle_s=0.8)
            print(f"sent P{pan} T{tilt} reply={reply or '<none>'}")
    finally:
        controller.close()