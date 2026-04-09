"""
Main Loop for Active Perception System.

Connects:
Camera -> Perception -> Uncertainty -> Policy

Logic:
1. MONITOR: Watch the scene. If uncertainty is high -> Switch to EXPLORE.
2. EXPLORE: Try different actions (exposure levels). Record which one is best.
3. STABILIZE: Apply the best action and go back to MONITOR.
"""

import cv2
import time
import numpy as np
from src.camera import Camera
from src.perception import PerceptionSystem
from src.uncertainty import UncertaintyEngine, TemporalSmoother
from src.policy import ActionPolicy


def _aruco_centroid_pixel(corners) -> tuple:
    """Mean (x, y) of the first marker's 4 corners in pixel coords."""
    if corners is None or len(corners) < 1:
        return None
    pts = corners[0][0]
    mx = float(np.mean(pts[:, 0]))
    my = float(np.mean(pts[:, 1]))
    return mx, my


class ActivePerceptionLoop:
    def __init__(self, camera_id: int = 1):
        print("Initializing System Modules...")
        
        # 1. Hardware
        # TIP: Change to 0 if using integrated webcam
        self.camera = Camera(camera_id)
        
        # 2. Perception & Brain
        self.perception = PerceptionSystem()
        self.uncertainty_engine = UncertaintyEngine()
        self.smoother = TemporalSmoother(window_size=5)
        
        # 3. Action
        self.policy = ActionPolicy(self.camera)
        
        # 4. System State
        self.state = "MONITOR"  # options: MONITOR, EXPLORE, EXPLORE_ZOOM
        self.current_exposure_idx = 3  # Start middle-ish index
        self.current_zoom_idx = 0  # Start with 1.0x
        
        # Exploration variables
        self.exploration_results = {} # {exposure_idx: average_uncertainty}
        self.explore_step = 0
        self.best_exposure_idx = 0
        self.zoom_exploration_results = {} # {zoom_idx: average_uncertainty}
        self.zoom_step = 0
        
        # Environmental Context
        self.baseline_brightness = None # To detect lighting changes
        self.brightness_change_ratio = 0.10 # 10% change triggers re-exploration
        self.frame_count = 0
        self.ignore_until_frame = 0 # Stabilization window
        self.zoom_ignore_until_frame = 0
        self.baseline_size = None
        self.size_change_ratio = 0.35
        self.size_change_floor = 200.0
        self.zoom_initialized = False

        # Session 9: Sniper Recovery / track (only when zoom > 1.0, state MONITOR)
        self._roi_lost_frames = 0
        self._roi_lost_threshold = 8
        self._sniper_recovery_active = False
        self._sniper_timeout_frames = 60 # Wait 2 seconds in wide-angle before giving up
        self._sniper_frame_count = 0
        
        # Initialize camera to default
        if self.policy.exposure_supported:
            self.policy.execute_exposure(self.current_exposure_idx)

    def run(self):#This is the main loop of the system
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
                
                # --- Step 3: Evaluate (Brain) ---
                raw_u, metrics = self.uncertainty_engine.compute(frame, corners)
                smooth_u = self.smoother.update(raw_u)
                
                # --- Step 4: Act (Decision Making) ---
                current_brightness = np.mean(frame)
                size_value = metrics.get("size_raw", 0.0)

                if self._sniper_recovery_active:
                    self._sniper_frame_count += 1
                    
                    if detected and corners is not None:
                        # Found it! Center ROI on target, but STAY at 1.0x.
                        # Reset initialization flags so the state machine re-optimizes zoom from scratch.
                        cent = _aruco_centroid_pixel(corners)
                        if cent is not None:
                            mx, my = cent
                            hh, ww = frame.shape[:2]
                            tnx, tny = self.policy.marker_center_to_full_norm(mx, my, ww, hh)
                            self.policy.set_roi_center(tnx, tny)
                            
                            print("[V] Sniper Locked! Target found at 1.0x. Triggering re-optimization.")
                            self._sniper_recovery_active = False
                            
                            # Force state machine to do a fresh EXPLORE_ZOOM
                            self.zoom_initialized = False
                            self.baseline_size = None
                            self.current_zoom_idx = 0
                            
                            # Stabilize before state machine starts reading garbage
                            self.ignore_until_frame = self.frame_count + 5
                            self.zoom_ignore_until_frame = self.frame_count + 5
                    
                    if self._sniper_recovery_active and self._sniper_frame_count >= self._sniper_timeout_frames:
                        # Timeout. Fallback to normal exploration.
                        print("[!] Sniper Timeout! Target completely lost. Calming down at 1.0x.")
                        self._sniper_recovery_active = False
                        
                        # Reset zoom initialization so that when it reappears, it triggers EXPLORE_ZOOM to re-optimize
                        self.zoom_initialized = False
                        self.baseline_size = None
                        self.current_zoom_idx = 0
                else:
                    if self.state == "MONITOR" and self.policy.current_zoom_level > 1.0:
                        if not detected:
                            self._roi_lost_frames += 1
                        else:
                            self._roi_lost_frames = 0
                    elif self.state != "MONITOR":
                        self._roi_lost_frames = 0

                    sniper_started = False
                    if (
                        self.state == "MONITOR"
                        and self.policy.current_zoom_level > 1.0
                        and self._roi_lost_frames >= self._roi_lost_threshold
                        and self.frame_count >= self.ignore_until_frame
                        and self.frame_count >= self.zoom_ignore_until_frame
                    ):
                        self._start_sniper_recovery()
                        sniper_started = True

                    if not sniper_started:
                        self._update_state_machine(smooth_u, current_brightness, size_value, detected)

                # Nudge ROI toward marker for next frame (zoomed + stable MONITOR only)
                if (
                    not self._sniper_recovery_active
                    and self.state == "MONITOR"
                    and self.policy.current_zoom_level > 1.0
                    and self.frame_count >= self.ignore_until_frame
                    and self.frame_count >= self.zoom_ignore_until_frame
                    and detected
                    and corners is not None
                ):
                    cent = _aruco_centroid_pixel(corners)
                    if cent is not None:
                        mx, my = cent
                        hh, ww = frame.shape[:2]
                        tnx, tny = self.policy.marker_center_to_full_norm(mx, my, ww, hh)
                        self.policy.nudge_roi_towards(tnx, tny, gain=0.15)
                
                # --- Step 5: Visualize ---
                vis_frame = self._draw_hud(frame, smooth_u, metrics, corners, ids)
                self.camera.display(vis_frame, "Active Perception System")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.release()
            print("System Shutdown.")

    def _cancel_sniper_recovery(self) -> None:
        if not self._sniper_recovery_active:
            return
        self._sniper_recovery_active = False
        self._sniper_frame_count = 0
        print("[i] Sniper Recovery cancelled")

    def _start_sniper_recovery(self) -> None:
        self._sniper_recovery_active = True
        self._sniper_frame_count = 0
        self._roi_lost_frames = 0
        
        # Action: Zoom out immediately to full view.
        print(f"[i] Sniper Recovery started! Target lost at {self.policy.current_zoom_level}x zoom.")
        self.policy.set_zoom(1.0)

    def _update_state_machine(self, uncertainty, current_brightness, size_value, detected):
        """
        Core Logic: Decides whether to stay monitoring or start exploring.
        """
        # Threshold to trigger exploration (e.g., if uncertainty > 0.6)
        TRIGGER_THRESHOLD = 0.6
        
        if self.state == "MONITOR":
            # 0. Stabilization Check
            if self.frame_count < self.ignore_until_frame:
                return

            # UPDATE BASELINE: If we just settled, capture the new "normal"
            if self.baseline_brightness is None:
                self.baseline_brightness = current_brightness
                print(f"[i] Baseline Brightness Set: {self.baseline_brightness:.1f}")

            # CHECK CHANGE: Has the environment changed significantly?
            env_changed = False
            if self.baseline_brightness is not None:
                diff = abs(current_brightness - self.baseline_brightness)
                
                # Dynamic Threshold: Ratio * Baseline (Weber's Law)
                # But keep a minimum floor (e.g. 5.0) to avoid noise in dark scenes
                dynamic_threshold = max(self.baseline_brightness * self.brightness_change_ratio, 5.0)
                
                if diff > dynamic_threshold:
                    env_changed = True
                    print(f"[!] Lighting Changed! Diff: {diff:.1f} baseline: {self.baseline_brightness:.1f} (Thresh: {dynamic_threshold:.1f})")

            # RULE: Only explore if confused AND (environment changed OR first run)
            # This prevents infinite loops when the best we can do is still bad.
            if uncertainty > TRIGGER_THRESHOLD:
                if env_changed:
                    print(f"[!] Triggering EXPLORE (Score: {uncertainty:.2f})")
                    self._cancel_sniper_recovery()
                    self.state = "EXPLORE"
                    self.explore_step = 0
                    self.exploration_results = {}
                    self.baseline_brightness = None # Reset baseline
                    return
            
            # --- Zoom exploration trigger (start or size change) ---
            if self.frame_count < self.zoom_ignore_until_frame:
                return
                
            if self.baseline_size is None and detected:
                self.baseline_size = size_value
                print(f"[i] Baseline Size Set: {self.baseline_size:.1f}")
            
            size_changed = False
            if self.baseline_size is not None and detected:
                size_diff = abs(size_value - self.baseline_size)
                size_threshold = max(self.baseline_size * self.size_change_ratio, self.size_change_floor)
                if size_diff > size_threshold:
                    size_changed = True
                    print(f"[!] Size Changed! Diff: {size_diff:.1f} baseline: {self.baseline_size:.1f} (Thresh: {size_threshold:.1f})")
            
            if (not self.zoom_initialized) or size_changed:
                print(f"[!] Triggering ZOOM EXPLORE (Score: {uncertainty:.2f})")
                self._cancel_sniper_recovery()
                self.state = "EXPLORE_ZOOM"
                self.zoom_step = 0
                self.zoom_exploration_results = {}
                self.baseline_size = None
                
        elif self.state == "EXPLORE":
            # In explore mode, we try one exposure per few frames
            # For simplicity in this demo, we assume 1 frame per step (fast sweep)
            # In reality, you might wait 5 frames for camera to settle.
            
            # 1. Record score for current setting
            current_idx = self.explore_step
            self.exploration_results[current_idx] = uncertainty
            print(f"   -> Testing Exp Level {current_idx}: Score {uncertainty:.2f}")
            
            # 2. Move to next step
            self.explore_step += 1
            
            # 3. Check if done
            if self.explore_step >= len(self.policy.exposure_levels):
                # Finished sweeping! Pick winner.
                self._apply_best_action()
                self.state = "MONITOR"
            else:
                # Execute next action
                self.policy.execute_exposure(self.explore_step)
                # Small sleep to let hardware settle
                time.sleep(0.1)
        
        elif self.state == "EXPLORE_ZOOM":
            # 1. Record score for current zoom setting
            current_idx = self.zoom_step
            self.zoom_exploration_results[current_idx] = uncertainty
            print(f"   -> Testing Zoom Level {current_idx}: Score {uncertainty:.2f}")
            
            # 2. Move to next step
            self.zoom_step += 1
            
            # 3. Check if done
            if self.zoom_step >= len(self.policy.zoom_levels):
                # Finished sweeping! Pick winner.
                self._apply_best_zoom()
                # After zoom is chosen, force an exposure adjustment sweep.
                # Rationale: zoom changes effective image content; re-tuning exposure can
                # improve detection robustness even if lighting didn't "change".
                if self.policy.exposure_supported:
                    print("[!] Triggering EXPLORE (Exposure) after ZOOM selection")
                    self.state = "EXPLORE"
                    self.explore_step = 0
                    self.exploration_results = {}
                    self.baseline_brightness = None
                    # Start the sweep at the first exposure level immediately so the
                    # next EXPLORE frame records the correct index=0 setting.
                    self.policy.execute_exposure(0)
                    time.sleep(0.1)
                else:
                    self.state = "MONITOR"
            else:
                # Execute next zoom action
                next_zoom = self.policy.zoom_levels[self.zoom_step]
                self.policy.set_zoom(next_zoom)

    def _apply_best_action(self):
        """Find the exposure index with lowest uncertainty."""
        if not self.exploration_results:
            return
            
        # 1. Find the minimum score
        min_score = min(self.exploration_results.values())
        
        # 2. Find all indices that have this score (or very close)
        candidates = [idx for idx, score in self.exploration_results.items() 
                     if abs(score - min_score) < 0.01]
                     
        # 3. Tie-breaking: Prefer a "good default" exposure.
        # Previously we preferred the brightest (highest index), which often picks -2
        # when multiple levels tie. For general cases this can be too bright, so we
        # prefer the candidate whose exposure value is closest to -4.
        preferred_exposure_val = -4
        best_idx = min(
            candidates,
            key=lambda idx: (
                abs(self.policy.exposure_levels[idx] - preferred_exposure_val),
                self.policy.exposure_levels[idx],  # if equally close, prefer darker (more negative)
            ),
        )
        best_score = self.exploration_results[best_idx]
        
        print(f"\n[V] Exploration Done. Winner: Level {best_idx} (Score {best_score:.2f})")
        
        self.policy.execute_exposure(best_idx)
        self.current_exposure_idx = best_idx
        
        # Reset baseline so MONITOR captures the new brightness as "Normal"
        self.baseline_brightness = None
        self.baseline_size = None  # Reset size baseline to avoid false zoom triggers after exposure changes
        self.ignore_until_frame = self.frame_count + 10 # Ignore 10 frames for camera settling

    def _apply_best_zoom(self):
        """Find the zoom index with lowest uncertainty."""
        if not self.zoom_exploration_results:
            return
            
        # 1. Find the minimum score
        min_score = min(self.zoom_exploration_results.values())
        
        # 2. Find all indices that have this score (or very close)
        candidates = [idx for idx, score in self.zoom_exploration_results.items() 
                     if abs(score - min_score) < 0.01]
                     
        # 3. Tie-breaking: Prefer lower zoom (wider FOV)
        best_idx = min(candidates)
        best_score = self.zoom_exploration_results[best_idx]
        
        print(f"\n[V] Zoom Exploration Done. Winner: Level {best_idx} (Score {best_score:.2f})")
        
        best_zoom = self.policy.zoom_levels[best_idx]
        self.policy.set_zoom(best_zoom)
        self.current_zoom_idx = best_idx
        self.zoom_initialized = True
        
        # Reset baseline so MONITOR captures the new size as "Normal"
        self.baseline_size = None
        self.zoom_ignore_until_frame = self.frame_count + 5

    def _draw_hud(self, frame, uncertainty, metrics, corners, ids):
        """
        Draw status on screen.
        Returns: Annotated frame
        """
        # 1. Draw markers
        # IMPORTANT: visualize returns a NEW image, capture it!
        annotated = self.perception.visualize(frame, corners, ids)
            
        # 2. Status Bar
        color = (0, 255, 0) if self.state == "MONITOR" else (0, 255, 255)
        mode = f"{self.state}+SNIPER" if self._sniper_recovery_active else self.state
        
        cv2.putText(annotated, f"MODE: {mode}", (220, 30), 
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
                   
        return annotated

if __name__ == "__main__":
    app = ActivePerceptionLoop()
    app.run()
