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

class ActivePerceptionLoop:
    def __init__(self):
        print("Initializing System Modules...")
        
        # 1. Hardware
        # TIP: Change to 0 if using integrated webcam
        self.camera = Camera(1)
        
        # 2. Perception & Brain
        self.perception = PerceptionSystem()
        self.uncertainty_engine = UncertaintyEngine()
        self.smoother = TemporalSmoother(window_size=5)
        
        # 3. Action
        self.policy = ActionPolicy(self.camera)
        
        # 4. System State
        self.state = "MONITOR"  # options: MONITOR, EXPLORE
        self.current_exposure_idx = 2  # Start middle-ish index
        
        # Exploration variables
        self.exploration_results = {} # {exposure_idx: average_uncertainty}
        self.explore_step = 0
        self.best_exposure_idx = 0
        
        # Environmental Context
        self.baseline_brightness = None # To detect lighting changes
        self.brightness_change_ratio = 0.10 # 20% change triggers re-exploration
        self.frame_count = 0
        self.ignore_until_frame = 0 # Stabilization window
        
        # Initialize camera to default
        if self.policy.exposure_supported:
            self.policy.execute_exposure(self.current_exposure_idx)

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
                detected, ids, corners = self.perception.detect(frame)
                
                # --- Step 3: Evaluate (Brain) ---
                raw_u, metrics = self.uncertainty_engine.compute(frame, corners)
                smooth_u = self.smoother.update(raw_u)
                
                # --- Step 4: Act (Decision Making) ---
                current_brightness = np.mean(frame)
                self._update_state_machine(smooth_u, current_brightness)
                
                # --- Step 5: Visualize ---
                vis_frame = self._draw_hud(frame, smooth_u, metrics, corners, ids)
                self.camera.display(vis_frame, "Active Perception System")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camera.release()
            print("System Shutdown.")

    def _update_state_machine(self, uncertainty, current_brightness):
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
                    self.state = "EXPLORE"
                    self.explore_step = 0
                    self.exploration_results = {}
                    self.baseline_brightness = None # Reset baseline
                
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

    def _apply_best_action(self):
        """Find the exposure index with lowest uncertainty."""
        if not self.exploration_results:
            return
            
        # 1. Find the minimum score
        min_score = min(self.exploration_results.values())
        
        # 2. Find all indices that have this score (or very close)
        candidates = [idx for idx, score in self.exploration_results.items() 
                     if abs(score - min_score) < 0.01]
                     
        # 3. Tie-breaking: Prefer higher exposure (e.g. -3 is better than -8)
        # Assuming higher index = higher exposure value in our policy list
        best_idx = max(candidates)
        best_score = self.exploration_results[best_idx]
        
        print(f"\n[V] Exploration Done. Winner: Level {best_idx} (Score {best_score:.2f})")
        
        self.policy.execute_exposure(best_idx)
        self.current_exposure_idx = best_idx
        
        # Reset baseline so MONITOR captures the new brightness as "Normal"
        self.baseline_brightness = None
        self.ignore_until_frame = self.frame_count + 10 # Ignore 10 frames for camera settling

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
        
        cv2.putText(annotated, f"MODE: {self.state}", (220, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 3. Uncertainty Bar
        bar_len = int(uncertainty * 200)
        u_color = (0, 0, 255) if uncertainty > 0.6 else (0, 255, 0)
        cv2.rectangle(annotated, (10, 50), (10 + bar_len, 70), u_color, -1)
        cv2.putText(annotated, f"Uncertainty: {uncertainty:.2f}", (220, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        return annotated

if __name__ == "__main__":
    app = ActivePerceptionLoop()
    app.run()
