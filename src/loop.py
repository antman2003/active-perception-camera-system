"""
Main Loop for Active Perception System.

Connects:
Camera -> Perception -> Uncertainty -> Policy

Refactored to use State Machine (Session 11).
"""

import cv2
import time
import numpy as np
from collections import deque
from src.camera import Camera
from src.controller import HardwareController
from src.logger import BlackboxLogger
from src.perception import PerceptionSystem
from src.uncertainty import UncertaintyEngine, TemporalSmoother
from src.policy import ActionPolicy
from src.states import MonitorState

class ActivePerceptionLoop:
    def __init__(
        self,
        camera_id: int = 1,
        debug: bool = False,
        enable_exposure_control: bool = True,
        enable_zoom_control: bool = True,
        enable_pan_tilt: bool = False,
        pan_tilt_port: str = "COM3",
        show_window: bool = True,
    ):
        print("Initializing System Modules...")
        
        # 1. Hardware
        self.camera = Camera(camera_id)
        
        self.enable_pan_tilt = enable_pan_tilt
        self.pan_tilt: HardwareController | None = None
        if enable_pan_tilt:
            try:
                self.pan_tilt = HardwareController(port=pan_tilt_port)
                self.pan_tilt.connect()
                print(f"[i] Pan-Tilt stage connected on {pan_tilt_port}")
            except Exception as e:
                print(f"[!] Pan-Tilt init failed ({e}). Continuing without physical servoing.")
                self.pan_tilt = None
                self.enable_pan_tilt = False
        
        # 2. Perception & Brain
        self.perception = PerceptionSystem()
        self.uncertainty_engine = UncertaintyEngine()
        self.smoother = TemporalSmoother(window_size=5)
        
        # 3. Action
        self.policy = ActionPolicy(self.camera)
        self.blackbox = BlackboxLogger(frame_logging_enabled=debug)
        self.policy.logger = self.blackbox
        self.enable_exposure_control = enable_exposure_control
        self.enable_zoom_control = enable_zoom_control
        self.show_window = show_window
        
        # 4. Context Variables (accessed by states)
        self.current_exposure_idx = 3
        self.current_zoom_idx = 0
        
        self.baseline_brightness = None
        self.brightness_change_ratio = 0.10
        self.frame_count = 0
        self.ignore_until_frame = 0
        self.zoom_ignore_until_frame = 0
        self.baseline_size = None
        self.size_change_ratio = 0.35
        self.size_change_floor = 200.0
        self.zoom_initialized = False

        # Stability and sweep parameters. Tune here instead of editing state logic.
        self.monitor_explore_enter_threshold = 0.60
        self.monitor_explore_exit_threshold = 0.50
        self.monitor_zoom_enter_threshold = 0.55
        self.monitor_zoom_exit_threshold = 0.45
        self.monitor_explore_trigger_frames = 2
        self.monitor_zoom_trigger_frames = 2
        self.monitor_roi_lost_threshold = 8
        self.monitor_nudge_gain = 0.15
        self.pan_tilt_gain_pan = 8.0
        self.pan_tilt_gain_tilt = 5.0
        self.pan_tilt_deadzone = 0.05
        self.pan_tilt_search_lost_threshold = 30
        self.last_seen_pan = None
        self.last_seen_tilt = None
        self.sniper_timeout_frames = 60

        self.exposure_settle_frames = 2
        self.exposure_sample_frames = 3
        self.zoom_settle_frames = 1
        self.zoom_sample_frames = 2

        # Detection debounce state.
        self.detect_confirm_frames = 2
        self.lost_confirm_frames = 3
        self.detected_streak = 0
        self.lost_streak = 0
        self.confirmed_detected = False
        self.confirmed_lost = True
        self.detection_history = deque(maxlen=30)
        self.benchmark_stable_window_frames = 10
        self.runtime_stats = {
            "frames": 0,
            "detected_frames": 0,
            "confirmed_detected_frames": 0,
            "monitor_frames": 0,
            "raw_u_sum": 0.0,
            "smooth_u_sum": 0.0,
            "size_sum": 0.0,
            "q_size_sum": 0.0,
            "q_sharpness_sum": 0.0,
            "detected_metric_frames": 0,
            "stable_samples": [],
        }
        
        # Initialize camera to default
        if self.policy.exposure_supported:
            self.policy.execute_exposure(self.current_exposure_idx)
            
        self.policy.set_zoom(self.policy.zoom_levels[self.current_zoom_idx])

        # 5. Initialize State Machine
        self.current_state = MonitorState()
        self.current_state.on_enter(self)
        self.blackbox.log_event(
            "session_started",
            camera_id=camera_id,
            initial_state=self.current_state.name,
        )

    def run(self, duration_s: float = None):
        print("\n=== Active Perception Loop Started ===")
        print("Press 'q' in the window to quit.")
        start_time = time.time()
        
        try:
            while True:
                if duration_s is not None and (time.time() - start_time) >= duration_s:
                    break

                self.frame_count += 1
                
                # --- Step 1: Sense ---
                ret, frame = self.camera.read()
                if not ret: break
                
                # --- Step 2: Perceive ---
                frame = self.policy.apply_digital_zoom(frame)
                detected, ids, corners = self.perception.detect(frame)
                if detected:
                    self.detected_streak += 1
                    self.lost_streak = 0
                else:
                    self.lost_streak += 1
                    self.detected_streak = 0

                self.confirmed_detected = self.detected_streak >= self.detect_confirm_frames
                self.confirmed_lost = self.lost_streak >= self.lost_confirm_frames
                self.detection_history.append(1 if detected else 0)
                
                # --- Step 3: Evaluate (Brain) ---
                raw_u, metrics = self.uncertainty_engine.compute(frame, corners)
                smooth_u = self.smoother.update(raw_u)
                
                # --- Step 4: Act (State Machine Update) ---
                current_brightness = np.mean(frame)
                size_value = metrics.get("size_raw", 0.0)
                detection_rate = sum(self.detection_history) / len(self.detection_history)
                elapsed_s = time.time() - start_time
                self._update_runtime_stats(raw_u, smooth_u, metrics, detected, elapsed_s)

                self.blackbox.log_frame(
                    frame_idx=self.frame_count,
                    state=self.current_state.name,
                    raw_u=raw_u,
                    smooth_u=smooth_u,
                    detected=detected,
                    confirmed_detected=self.confirmed_detected,
                    confirmed_lost=self.confirmed_lost,
                    detection_rate=detection_rate,
                    brightness=float(current_brightness),
                    zoom=float(self.policy.current_zoom_level),
                    exposure_idx=self.current_exposure_idx,
                    metrics=metrics,
                )

                next_state = self.current_state.update(
                    self, frame, detected, corners, ids, 
                    smooth_u, raw_u, metrics, current_brightness, size_value
                )

                if next_state != self.current_state:
                    self.blackbox.log_event(
                        "state_transition",
                        frame_idx=self.frame_count,
                        from_state=self.current_state.name,
                        to_state=next_state.name,
                        raw_u=raw_u,
                        smooth_u=smooth_u,
                        detected=detected,
                        confirmed_detected=self.confirmed_detected,
                        confirmed_lost=self.confirmed_lost,
                        zoom=float(self.policy.current_zoom_level),
                    )
                    self.current_state.on_exit(self)
                    self.current_state = next_state
                    self.current_state.on_enter(self)

                # --- Step 5: Visualize ---
                if self.show_window:
                    vis_frame = self._draw_hud(frame, smooth_u, metrics, corners, ids)
                    self.camera.display(vis_frame, "Active Perception System")
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
        finally:
            self.blackbox.log_event(
                "session_stopped",
                final_state=self.current_state.name,
                frame_idx=self.frame_count,
            )
            if self.pan_tilt is not None:
                try:
                    self.pan_tilt.home(smooth=True)
                except Exception:
                    pass
                self.pan_tilt.close()
            self.camera.release()
            print("System Shutdown.")

        return self.build_summary(time.time() - start_time)

    def _update_runtime_stats(self, raw_u, smooth_u, metrics, detected, elapsed_s: float):
        self.runtime_stats["frames"] += 1
        self.runtime_stats["raw_u_sum"] += float(raw_u)
        self.runtime_stats["smooth_u_sum"] += float(smooth_u)
        if detected:
            self.runtime_stats["detected_frames"] += 1
        if self.confirmed_detected:
            self.runtime_stats["confirmed_detected_frames"] += 1
        if self.current_state.name == "MONITOR":
            self.runtime_stats["monitor_frames"] += 1
        if metrics.get("detected", False):
            self.runtime_stats["detected_metric_frames"] += 1
            self.runtime_stats["size_sum"] += float(metrics.get("size_raw", 0.0))
            self.runtime_stats["q_size_sum"] += float(metrics.get("q_size", 0.0))
            self.runtime_stats["q_sharpness_sum"] += float(metrics.get("q_sharpness", 0.0))

        is_stable_monitor_frame = (
            self.current_state.name == "MONITOR"
            and self.confirmed_detected
            and self.frame_count >= self.ignore_until_frame
            and self.frame_count >= self.zoom_ignore_until_frame
        )
        if is_stable_monitor_frame:
            self.runtime_stats["stable_samples"].append(
                {
                    "frame_idx": int(self.frame_count),
                    "elapsed_s": float(elapsed_s),
                    "raw_u": float(raw_u),
                    "smooth_u": float(smooth_u),
                    "q_size": float(metrics.get("q_size", 0.0)),
                    "q_sharpness": float(metrics.get("q_sharpness", 0.0)),
                    "size_raw": float(metrics.get("size_raw", 0.0)),
                    "zoom": float(self.policy.current_zoom_level),
                    "exposure_idx": int(self.current_exposure_idx),
                }
            )

    def build_summary(self, duration_s: float) -> dict:
        frames = max(1, self.runtime_stats["frames"])
        detected_metric_frames = max(1, self.runtime_stats["detected_metric_frames"])
        stable_metrics = self._compute_stable_metrics()
        return {
            "duration_s": float(duration_s),
            "frames": self.runtime_stats["frames"],
            "detected_rate": self.runtime_stats["detected_frames"] / frames,
            "confirmed_detected_rate": self.runtime_stats["confirmed_detected_frames"] / frames,
            "search_frames_ratio": 1.0 - (self.runtime_stats["monitor_frames"] / frames),
            "avg_raw_uncertainty": self.runtime_stats["raw_u_sum"] / frames,
            "avg_smooth_uncertainty": self.runtime_stats["smooth_u_sum"] / frames,
            "avg_size_when_detected": self.runtime_stats["size_sum"] / detected_metric_frames,
            "avg_q_size_when_detected": self.runtime_stats["q_size_sum"] / detected_metric_frames,
            "avg_q_sharpness_when_detected": self.runtime_stats["q_sharpness_sum"] / detected_metric_frames,
            "final_zoom": float(self.policy.current_zoom_level),
            "final_exposure_idx": int(self.current_exposure_idx),
            **stable_metrics,
            "mode_flags": {
                "enable_exposure_control": self.enable_exposure_control,
                "enable_zoom_control": self.enable_zoom_control,
            },
        }

    def _compute_stable_metrics(self) -> dict:
        samples = self.runtime_stats["stable_samples"]
        window = self.benchmark_stable_window_frames
        if len(samples) < window:
            return {
                "stable_window_frames": window,
                "stable_samples": len(samples),
                "time_to_stable_s": None,
                "best_stable_uncertainty": None,
                "best_stable_smooth_uncertainty": None,
                "best_stable_q_size": None,
                "best_stable_q_sharpness": None,
                "best_stable_size_raw": None,
                "best_stable_zoom": None,
                "best_stable_exposure_idx": None,
                "final_stable_uncertainty": None,
                "final_stable_smooth_uncertainty": None,
                "final_stable_q_size": None,
                "final_stable_q_sharpness": None,
                "final_stable_size_raw": None,
                "final_stable_zoom": None,
                "final_stable_exposure_idx": None,
            }

        stable_runs = []
        current_run = [samples[0]]
        for sample in samples[1:]:
            if sample["frame_idx"] == current_run[-1]["frame_idx"] + 1:
                current_run.append(sample)
            else:
                stable_runs.append(current_run)
                current_run = [sample]
        stable_runs.append(current_run)

        window_summaries = []
        for run in stable_runs:
            if len(run) < window:
                continue
            for i in range(len(run) - window + 1):
                chunk = run[i : i + window]
                window_summaries.append(
                    {
                        "elapsed_s": chunk[-1]["elapsed_s"],
                        "raw_u": float(np.mean([s["raw_u"] for s in chunk])),
                        "smooth_u": float(np.mean([s["smooth_u"] for s in chunk])),
                        "q_size": float(np.mean([s["q_size"] for s in chunk])),
                        "q_sharpness": float(np.mean([s["q_sharpness"] for s in chunk])),
                        "size_raw": float(np.mean([s["size_raw"] for s in chunk])),
                        "zoom": float(np.mean([s["zoom"] for s in chunk])),
                        "exposure_idx": int(round(np.mean([s["exposure_idx"] for s in chunk]))),
                    }
                )

        if not window_summaries:
            return {
                "stable_window_frames": window,
                "stable_samples": len(samples),
                "time_to_stable_s": None,
                "best_stable_uncertainty": None,
                "best_stable_smooth_uncertainty": None,
                "best_stable_q_size": None,
                "best_stable_q_sharpness": None,
                "best_stable_size_raw": None,
                "best_stable_zoom": None,
                "best_stable_exposure_idx": None,
                "final_stable_uncertainty": None,
                "final_stable_smooth_uncertainty": None,
                "final_stable_q_size": None,
                "final_stable_q_sharpness": None,
                "final_stable_size_raw": None,
                "final_stable_zoom": None,
                "final_stable_exposure_idx": None,
            }

        best_window = min(window_summaries, key=lambda item: item["raw_u"])
        final_window = window_summaries[-1]

        return {
            "stable_window_frames": window,
            "stable_samples": len(samples),
            "time_to_stable_s": window_summaries[0]["elapsed_s"],
            "best_stable_uncertainty": best_window["raw_u"],
            "best_stable_smooth_uncertainty": best_window["smooth_u"],
            "best_stable_q_size": best_window["q_size"],
            "best_stable_q_sharpness": best_window["q_sharpness"],
            "best_stable_size_raw": best_window["size_raw"],
            "best_stable_zoom": best_window["zoom"],
            "best_stable_exposure_idx": best_window["exposure_idx"],
            "final_stable_uncertainty": final_window["raw_u"],
            "final_stable_smooth_uncertainty": final_window["smooth_u"],
            "final_stable_q_size": final_window["q_size"],
            "final_stable_q_sharpness": final_window["q_sharpness"],
            "final_stable_size_raw": final_window["size_raw"],
            "final_stable_zoom": final_window["zoom"],
            "final_stable_exposure_idx": final_window["exposure_idx"],
        }

    def _draw_hud(self, frame, uncertainty, metrics, corners, ids):
        """
        Draw status on screen.
        Returns: Annotated frame
        """
        # 1. Draw markers
        annotated = self.perception.visualize(frame, corners, ids)
            
        # 2. Status Bar
        color = (0, 255, 0) if self.current_state.name == "MONITOR" else (0, 255, 255)
        
        cv2.putText(annotated, f"MODE: {self.current_state.name}", (220, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 3. Uncertainty Bar
        bar_len = int(uncertainty * 200)
        u_color = (0, 0, 255) if uncertainty > 0.6 else (0, 255, 0)
        cv2.rectangle(annotated, (10, 50), (10 + bar_len, 70), u_color, -1)
        cv2.putText(annotated, f"Uncertainty: {uncertainty:.2f}", (220, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        # 4. Zoom Level HUD
        zoom_text = f"ZOOM: {self.policy.current_zoom_level}x"
        cv2.putText(annotated, zoom_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        # 5. Debug Info (Sharpness & Size)
        cv2.putText(annotated, f"Sharpness: {metrics.get('sharpness_raw', 0):.0f}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated, f"Size: {metrics.get('size_raw', 0):.0f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                   
        # 6. Pan-Tilt pose
        if self.pan_tilt is not None:
            pose = self.pan_tilt.current_pose
            cv2.putText(annotated, f"PT: P{pose.pan} T{pose.tilt}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
                   
        return annotated

if __name__ == "__main__":
    app = ActivePerceptionLoop()
    app.run()
