"""
Serial protocol + safe servo command API for pan/tilt servos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import serial


@dataclass(frozen=True)
class PanTiltPose:
    """Logical pose command for the stage."""

    pan: int
    tilt: int


@dataclass(frozen=True)
class MotionProfile:
    """Shared motion pacing settings."""

    step_deg: int = 10
    step_settle_s: float = 0.25
    settle_s: float = 0.3


@dataclass(frozen=True)
class HardwareConfig:
    """Controller configuration that can be reused by future runtime code."""

    port: str = "COM3"
    baudrate: int = 115200
    pan_limits: Tuple[int, int] = (0, 180)
    tilt_limits: Tuple[int, int] = (50, 130)
    home_pose: PanTiltPose = PanTiltPose(90, 90)
    startup_delay_s: float = 2.0
    timeout_s: float = 1.0
    motion: MotionProfile = MotionProfile()


class HardwareController:
    """
    Hardware abstraction layer for the pan-tilt stage.

    Expected command format:
        P90 T90
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        pan_limits: Tuple[int, int] = (0, 180),
        tilt_limits: Tuple[int, int] = (50, 130),
        home_pose: Tuple[int, int] = (90, 90),
        startup_delay_s: float = 2.0,
        timeout_s: float = 1.0,
        motion_profile: MotionProfile = MotionProfile(),
    ):
        self.config = HardwareConfig(
            port=port,
            baudrate=baudrate,
            pan_limits=pan_limits,
            tilt_limits=tilt_limits,
            home_pose=PanTiltPose(*home_pose),
            startup_delay_s=startup_delay_s,
            timeout_s=timeout_s,
            motion=motion_profile,
        )
        self.port = self.config.port
        self.baudrate = self.config.baudrate
        self.pan_limits = self.config.pan_limits
        self.tilt_limits = self.config.tilt_limits
        self.startup_delay_s = self.config.startup_delay_s
        self.timeout_s = self.config.timeout_s
        self.home_pose = self.config.home_pose
        self.motion_profile = self.config.motion

        self.serial_port: Optional[serial.Serial] = None
        self.current_pan = self.home_pose.pan
        self.current_tilt = self.home_pose.tilt

    @property
    def is_connected(self) -> bool:
        return self.serial_port is not None and self.serial_port.is_open

    @property
    def current_pose(self) -> PanTiltPose:
        return PanTiltPose(self.current_pan, self.current_tilt)

    def __enter__(self) -> "HardwareController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        """Open the serial port and give the board time to reboot."""
        if self.is_connected:
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
        if self.is_connected:
            self.serial_port.close()
        self.serial_port = None

    def _require_connection(self) -> serial.Serial:
        if not self.is_connected:
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

    def clamp_pose(self, pan: Optional[int] = None, tilt: Optional[int] = None) -> PanTiltPose:
        """Clamp a target pose to the configured software limits."""
        target_pan = self.current_pan if pan is None else self._clamp(pan, self.pan_limits)
        target_tilt = self.current_tilt if tilt is None else self._clamp(tilt, self.tilt_limits)
        return PanTiltPose(target_pan, target_tilt)

    def send_ping(self) -> str:
        """Check whether the board is alive on the serial link."""
        port = self._require_connection()
        port.write(b"PING\n")
        port.flush()
        time.sleep(0.1)
        return self._drain_input()

    def _send_pose(
        self,
        pose: PanTiltPose,
        settle_s: float = 0.2,
    ) -> str:
        """
        Send one clamped pan/tilt command.

        Returns any serial response captured shortly after the command.
        """
        port = self._require_connection()
        cmd = f"P{pose.pan} T{pose.tilt}\n"
        port.write(cmd.encode("utf-8"))
        port.flush()
        time.sleep(settle_s)

        self.current_pan = pose.pan
        self.current_tilt = pose.tilt
        return self._drain_input()

    def move_to(
        self,
        pan: Optional[int] = None,
        tilt: Optional[int] = None,
        settle_s: Optional[float] = None,
        smooth: bool = False,
        step_deg: Optional[int] = None,
        step_settle_s: Optional[float] = None,
    ) -> str:
        """Move to an absolute pose."""
        target_pose = self.clamp_pose(pan, tilt)
        if smooth:
            effective_step_settle_s = (
                step_settle_s
                if step_settle_s is not None
                else settle_s
                if settle_s is not None
                else self.motion_profile.step_settle_s
            )
            return self.move_smoothly(
                target_pose.pan,
                target_pose.tilt,
                step_deg=step_deg or self.motion_profile.step_deg,
                step_settle_s=effective_step_settle_s,
            )
        return self._send_pose(target_pose, settle_s=settle_s or self.motion_profile.settle_s)

    def move_by(
        self,
        delta_pan: float = 0.0,
        delta_tilt: float = 0.0,
        smooth: bool = False,
        settle_s: Optional[float] = None,
        step_deg: Optional[int] = None,
        step_settle_s: Optional[float] = None,
    ) -> str:
        """Move relative to the current pose."""
        next_pan = round(self.current_pan + delta_pan)
        next_tilt = round(self.current_tilt + delta_tilt)
        return self.move_to(
            next_pan,
            next_tilt,
            smooth=smooth,
            settle_s=settle_s,
            step_deg=step_deg,
            step_settle_s=step_settle_s,
        )

    def set_pan_tilt(
        self,
        pan: Optional[int] = None,
        tilt: Optional[int] = None,
        settle_s: float = 0.2,
    ) -> str:
        """Backwards-compatible absolute move helper."""
        return self.move_to(pan, tilt, settle_s=settle_s)

    def center(self, settle_s: float = 0.3) -> str:
        return self.home(settle_s=settle_s)

    def home(self, smooth: bool = True, settle_s: Optional[float] = None) -> str:
        """Move back to the configured home pose."""
        return self.move_to(
            self.home_pose.pan,
            self.home_pose.tilt,
            smooth=smooth,
            settle_s=settle_s,
        )

    def nudge(self, delta_pan: float = 0.0, delta_tilt: float = 0.0, settle_s: float = 0.15) -> str:
        return self.move_by(delta_pan, delta_tilt, settle_s=settle_s)

    def move_smoothly(
        self,
        pan: Optional[int] = None,
        tilt: Optional[int] = None,
        step_deg: int = 10,
        step_settle_s: float = 0.1,
    ) -> str:
        """
        Move to the target pose in smaller increments.

        This is more reliable for heavier pan-tilt stages because it gives the
        servos time to physically catch up between larger target changes.
        """
        target_pose = self.clamp_pose(pan, tilt)
        target_pan = target_pose.pan
        target_tilt = target_pose.tilt

        start_pan = self.current_pan
        start_tilt = self.current_tilt
        delta_pan = target_pan - start_pan
        delta_tilt = target_tilt - start_tilt
        max_delta = max(abs(delta_pan), abs(delta_tilt))

        if max_delta == 0:
            return self._drain_input()

        step_deg = max(1, int(step_deg))
        num_steps = max(1, (max_delta + step_deg - 1) // step_deg)

        reply = ""
        for idx in range(1, num_steps + 1):
            next_pan = round(start_pan + (delta_pan * idx) / num_steps)
            next_tilt = round(start_tilt + (delta_tilt * idx) / num_steps)
            reply = self._send_pose(PanTiltPose(next_pan, next_tilt), settle_s=step_settle_s)
        return reply

    def sweep_poses(
        self,
        poses: Iterable[PanTiltPose],
        smooth: bool = True,
        step_deg: Optional[int] = None,
        step_settle_s: Optional[float] = None,
        settle_s: Optional[float] = None,
    ) -> list[str]:
        """Execute a sequence of poses and collect board replies."""
        replies = []
        for pose in poses:
            reply = self.move_to(
                pose.pan,
                pose.tilt,
                smooth=smooth,
                settle_s=settle_s,
                step_deg=step_deg,
                step_settle_s=step_settle_s,
            )
            replies.append(reply)
        return replies


def run_servo_demo(
    port: str = "COM3",
    baudrate: int = 115200, #波特率
    pan_limits: Tuple[int, int] = (0, 180),
    tilt_limits: Tuple[int, int] = (50, 130),
) -> None:
    """
    Send a conservative motion sequence using stepped smooth moves.
    """
    with HardwareController(
        port=port,
        baudrate=baudrate,
        pan_limits=pan_limits,
        tilt_limits=tilt_limits,
        motion_profile=MotionProfile(step_deg=10, step_settle_s=0.3, settle_s=0.3),
    ) as controller:
        sequence = [
            controller.home_pose,
            PanTiltPose(pan_limits[0], controller.home_pose.tilt),
            PanTiltPose(pan_limits[1], controller.home_pose.tilt),
            controller.home_pose,
            PanTiltPose(controller.home_pose.pan, tilt_limits[0]),
            PanTiltPose(controller.home_pose.pan, tilt_limits[1]),
            controller.home_pose,
        ]
        replies = controller.sweep_poses(sequence, smooth=True)
        for pose, reply in zip(sequence, replies):
            print(f"sent P{pose.pan} T{pose.tilt} reply={reply or '<none>'}")


def run_interactive_stage_control(
    port: str = "COM3",
    baudrate: int = 115200,
    pan_limits: Tuple[int, int] = (15, 165),
    tilt_limits: Tuple[int, int] = (50, 130),
    settle_s: float = 0.25,
) -> None:
    """
    Interactive terminal loop for manually controlling pan/tilt.

    Commands:
    - "<pan> <tilt>" to move both axes, e.g. "90 100"
    - "p <pan>" to update pan only, e.g. "p 45"
    - "t <tilt>" to update tilt only, e.g. "t 110"
    - "c" to center at 90, 90
    - "q" to quit
    """
    with HardwareController(
        port=port,
        baudrate=baudrate,
        pan_limits=pan_limits,
        tilt_limits=tilt_limits,
        motion_profile=MotionProfile(step_deg=10, step_settle_s=settle_s, settle_s=settle_s),
    ) as controller:
        print("Interactive pan-tilt control")
        print(f"Pan limits: {pan_limits[0]} to {pan_limits[1]}")
        print(f"Tilt limits: {tilt_limits[0]} to {tilt_limits[1]}")
        print('Enter "<pan> <tilt>", "p <pan>", "t <tilt>", "c", or "q".')

        while True:
            raw = input("pan-tilt> ").strip()
            if not raw:
                continue

            lowered = raw.lower()
            if lowered in {"q", "quit", "exit"}:
                break

            try:
                if lowered in {"c", "center"}:
                    reply = controller.home(smooth=True, settle_s=settle_s)
                    print(
                        f"sent P{controller.current_pan} T{controller.current_tilt} "
                        f"reply={reply or '<none>'}"
                    )
                    continue

                parts = raw.split()
                if len(parts) == 2 and parts[0].lower() == "p":
                    target_pan = int(parts[1])
                    reply = controller.move_to(target_pan, None, smooth=True, step_settle_s=settle_s)
                    print(
                        f"sent P{controller.current_pan} T{controller.current_tilt} "
                        f"reply={reply or '<none>'}"
                    )
                    continue

                if len(parts) == 2 and parts[0].lower() == "t":
                    target_tilt = int(parts[1])
                    reply = controller.move_to(None, target_tilt, smooth=True, step_settle_s=settle_s)
                    print(
                        f"sent P{controller.current_pan} T{controller.current_tilt} "
                        f"reply={reply or '<none>'}"
                    )
                    continue

                if len(parts) == 2:
                    target_pan = int(parts[0])
                    target_tilt = int(parts[1])
                    reply = controller.move_to(target_pan, target_tilt, smooth=True, step_settle_s=settle_s)
                    print(
                        f"sent P{controller.current_pan} T{controller.current_tilt} "
                        f"reply={reply or '<none>'}"
                    )
                    continue

                print('Invalid input. Use "<pan> <tilt>", "p <pan>", "t <tilt>", "c", or "q".')
            except ValueError:
                print("Angles must be integers.")