"""
Uncertainty Estimation Module.

This module acts as the "Brain" of the perception system.
It evaluates HOW reliable the current perception result is.

Metrics used:
1. Detection Status: Found or not?
2. Sharpness: Is the image blurry? (Laplacian Variance)
3. Target Size: Is the marker too small? (Pixel Area)

It also includes Temporal Smoothing to prevent jittery decisions.
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, List, Optional

class UncertaintyEngine:
    """
    Computes "Uncertainty Score" (0.0 to 1.0) for a single frame.
    0.0 = Perfect confidence
    1.0 = Total uncertainty (blind)
    """
    
    def __init__(self, 
                 sharpness_low=20.0, sharpness_high=300.0,
                 size_low=800.0, size_high=100000):
        """
        Args:
            sharpness_low/high: Thresholds for Laplacian variance.
            size_low/high: Thresholds for marker pixel area.
        """
        self.s_low = sharpness_low
        self.s_high = sharpness_high
        self.a_low = size_low
        self.a_high = size_high

    def compute(self, frame: np.ndarray, corners: list) -> Tuple[float, dict]:
        """
        Compute uncertainty score.
        
        Returns:
            score (float): 0.0 ~ 1.0
            metrics (dict): Raw values for debugging (sharpness, area, etc.)
        """
        # 1. Compute Raw Metrics
        sharpness_val = self._compute_sharpness(frame)
        
        detected = (corners is not None and len(corners) > 0)
        area_val = 0.0
        if detected:
            # Use the area of the first marker
            area_val = cv2.contourArea(corners[0])

        # 2. Normalize to Quality (0.0=Bad, 1.0=Good)
        q_sharpness = self._normalize(sharpness_val, self.s_low, self.s_high)
        q_size = self._normalize(area_val, self.a_low, self.a_high)

        # 3. Compute Uncertainty Logic
        if not detected:
            # Case A: Nothing found -> High Uncertainty
            # We don't say 1.0 immediately to allow 'flicker' recovery, 
            # but usually it's high. Let's say 0.9.
            score = 0.9
        else:
            # Case B: Found, but how good is it?
            # Base uncertainty = 0.1 (nothing is perfect)
            # Penalty for blur: up to 0.4
            # Penalty for small size: up to 0.4
            
            penalty_blur = (1.0 - q_sharpness) * 0.4
            penalty_size = (1.0 - q_size) * 0.4
            
            score = 0.1 + penalty_blur + penalty_size

        # Clamp result to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        
        metrics = {
            "detected": detected,
            "sharpness_raw": sharpness_val,
            "size_raw": area_val,
            "q_sharpness": q_sharpness,
            "q_size": q_size
        }
        
        return score, metrics

    def _compute_sharpness(self, frame: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian Variance.
        Higher = Sharper.
        """
        if frame is None:
            return 0.0
        # Convert to gray if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _normalize(self, value, low, high) -> float:
        """Map value to 0.0-1.0 range based on thresholds."""
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)


class TemporalSmoother:
    """
    Smoothing wrapper to stabilize outputs over time.
    """
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
    
    def update(self, new_value: float) -> float:
        """Add new value and return moving average."""
        self.history.append(new_value)
        return sum(self.history) / len(self.history)


# --- Independent Test ---
def main():
    from src.camera import Camera
    from src.perception import PerceptionSystem
    import time
    
    print("Initializing modules...")
    camera = Camera(1)
    perception = PerceptionSystem()
    uncertainty_engine = UncertaintyEngine()
    smoother = TemporalSmoother(window_size=5)
    
    print("Running... (Press 'q' in window to quit)")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret: break
            
            # 1. Perception
            detected, ids, corners = perception.detect(frame)
            
            # 2. Uncertainty
            raw_score, metrics = uncertainty_engine.compute(frame, corners)
            
            # 3. Smoothing
            smooth_score = smoother.update(raw_score)
            
            # 4. Visualization
            vis_frame = perception.visualize(frame, corners, ids)
            
            # Draw HUD
            # Bar chart for uncertainty (Red=High, Green=Low)
            bar_width = int(smooth_score * 200)
            color = (0, 0, 255) if smooth_score > 0.5 else (0, 255, 0)
            cv2.rectangle(vis_frame, (10, 60), (10 + bar_width, 80), color, -1)
            cv2.putText(vis_frame, f"Uncertainty: {smooth_score:.2f}", (220, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Debug info
            cv2.putText(vis_frame, f"Sharpness: {metrics['sharpness_raw']:.0f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(vis_frame, f"Size: {metrics['size_raw']:.0f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            camera.display(vis_frame, "Uncertainty Test")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        print("Done.")

if __name__ == "__main__":
    main()
