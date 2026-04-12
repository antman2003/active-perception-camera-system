"""
Main Loop for Active Perception System.

Connects:
Camera -> Perception -> Uncertainty -> Policy

Refactored to use State Machine (Session 11).
"""

import cv2
import time
import numpy as np
from src.camera import Camera
from src.perception import PerceptionSystem
from src.uncertainty import UncertaintyEngine, TemporalSmoother
from src.policy import ActionPolicy
from src.states import MonitorState

class ActivePerceptionLoop:
    def __init__(self, camera_id: int = 1):
        print("Initializing System Modules...")
        
        # 1. Hardware
        self.camera = Camera(camera_id)
        
        # 2. Perception & Brain
        self.perception = PerceptionSystem()
        self.uncertainty_engine = UncertaintyEngine()
        self.smoother = TemporalSmoother(window_size=5)
        
        # 3. Action
        self.policy = ActionPolicy(self.camera)
        
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
        
        # Initialize camera to default
        if self.policy.exposure_supported:
            self.policy.execute_exposure(self.current_exposure_idx)
            
        self.policy.set_zoom(self.policy.zoom_levels[self.current_zoom_idx])

        # 5. Initialize State Machine
        self.current_state = MonitorState()
        self.current_state.on_enter(self)

    def run(self):
        print("\n=== Active Perception Loop Started ===")
        print("Press 'q' in the window to quit.")
        
        try:
            while True:
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
                
                # --- Step 3: Evaluate (Brain) ---
                raw_u, metrics = self.uncertainty_engine.compute(frame, corners)
                smooth_u = self.smoother.update(raw_u)
                
                # --- Step 4: Act (State Machine Update) ---
                current_brightness = np.mean(frame)
                size_value = metrics.get("size_raw", 0.0)

                next_state = self.current_state.update(
                    self, frame, detected, corners, ids, 
                    smooth_u, raw_u, metrics, current_brightness, size_value
                )

                if next_state != self.current_state:
                    self.current_state.on_exit(self)
                    self.current_state = next_state
                    self.current_state.on_enter(self)

                # --- Step 5: Visualize ---
                vis_frame = self._draw_hud(frame, smooth_u, metrics, corners, ids)
                self.camera.display(vis_frame, "Active Perception System")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.release()
            print("System Shutdown.")

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
                   
        return annotated

if __name__ == "__main__":
    app = ActivePerceptionLoop()
    app.run()
