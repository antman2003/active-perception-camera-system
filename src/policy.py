"""
Policy module for active perception system.

Defines the "Action Space" (what the system can DO).
Currently supports:
1. Exposure Control (Physical)
"""

import cv2
import time
from typing import List, Optional

class ActionPolicy:
    """
    Manages camera actions (Exposure, etc.).
    """
    
    def __init__(self, camera):
        """
        Initialize policy and check hardware capabilities.
        """
        self.camera = camera
        self.logger = None
        self.exposure_supported = self._check_exposure_support()
        
        # Define the "Action Space" (Discrete levels)
        
        # 1. Exposure Levels (Hardware)
        # Note: OpenCV exposure values are usually log2 seconds.
        # -1 = 640ms, -2 = 320ms ... -5 = 40ms, -6 = 20ms, -7 = 10ms
        self.exposure_levels = [-8, -7, -6, -5, -4, -3, -2]
        
        # 2. Zoom Levels (Software / Digital)
        # 1.0 = No Zoom (Full Frame)
        # 2.0 = 2x Zoom (Center 50% Area)
        # 4.0 = 4x Zoom (Center 25% Area)
        self.zoom_levels = [1.0, 1.5, 2.0, 3.0,4.0]
        self.current_zoom_level = 1.0 # Default start

        # ROI center on the full input frame (normalized 0..1). Default = image center.
        # Only used when current_zoom_level > 1.0.
        self._roi_nx = 0.5
        self._roi_ny = 0.5

    def reset_roi_center(self) -> None:
        """Reset digital ROI to the image center (1.0x zoom behavior)."""
        self._roi_nx = 0.5
        self._roi_ny = 0.5

    def _roi_norm_bounds_for_zoom(self, zoom_level: float) -> tuple:
        """Valid (lo, hi) for normalized ROI center at a given zoom."""
        z = zoom_level
        if z <= 1.0:
            return 0.5, 0.5
        lo = 1.0 / (2.0 * z)
        hi = 1.0 - lo
        return lo, hi

    def _roi_norm_bounds(self) -> tuple:
        """Valid (lo, hi) for normalized ROI center at current zoom (prevents crop overflow)."""
        return self._roi_norm_bounds_for_zoom(self.current_zoom_level)

    def _clamp_roi_norm(self, nx: float, ny: float, zoom_level: float = None) -> tuple:
        z = self.current_zoom_level if zoom_level is None else zoom_level
        if z <= 1.0:
            return 0.5, 0.5
        lo, hi = self._roi_norm_bounds_for_zoom(z)
        return max(lo, min(hi, nx)), max(lo, min(hi, ny))

    def set_roi_center(self, nx: float, ny: float, zoom_level: float = None) -> None:
        """Set ROI center in normalized full-frame coordinates."""
        z = self.current_zoom_level if zoom_level is None else zoom_level
        if z <= 1.0:
            self.reset_roi_center()
            return
        self._roi_nx, self._roi_ny = self._clamp_roi_norm(nx, ny, zoom_level=z)

    def nudge_roi_towards(self, nx: float, ny: float, gain: float = 0.15) -> None:
        """Move ROI center one step toward a target (e.g. marker) in normalized full-frame coords."""
        if self.current_zoom_level <= 1.0:
            return
        tx, ty = self._clamp_roi_norm(nx, ny)
        self._roi_nx += gain * (tx - self._roi_nx)
        self._roi_ny += gain * (ty - self._roi_ny)
        self._roi_nx, self._roi_ny = self._clamp_roi_norm(self._roi_nx, self._roi_ny)

    def _digital_zoom_crop_rect(self, w: int, h: int) -> tuple:
        """
        Crop rectangle on the full frame for the current zoom + ROI.
        Matches apply_digital_zoom geometry (integer crop size, top-left clamped to image).
        """
        z = self.current_zoom_level
        new_w = int(w / z)
        new_h = int(h / z)
        cx = int(self._roi_nx * w)
        cy = int(self._roi_ny * h)
        x1 = cx - new_w // 2
        y1 = cy - new_h // 2
        x1 = max(0, min(x1, w - new_w))
        y1 = max(0, min(y1, h - new_h))
        return x1, y1, new_w, new_h

    def marker_center_to_full_norm(self, mx: float, my: float, w: int, h: int) -> tuple:
        """
        Map marker center (pixels) in the zoomed output image back to normalized coords on the full input frame.
        """
        if self.current_zoom_level <= 1.0:
            return mx / w, my / h
        x1, y1, new_w, new_h = self._digital_zoom_crop_rect(w, h)
        sx = mx * new_w / float(w)
        sy = my * new_h / float(h)
        full_x = x1 + sx
        full_y = y1 + sy
        return full_x / w, full_y / h

    def apply_digital_zoom(self, frame):
        """
        Apply current zoom level to the frame (Center Crop & Resize).
        
        Args:
            frame: Original captured frame.
            
        Returns:
            Zoomed frame (same resolution as original).
        """
        if self.current_zoom_level == 1.0:
            return frame
            
        h, w = frame.shape[:2]
        
        # Calculate crop box size
        # If zoom is 2.0, we want 1/2 the width and height
        x1, y1, new_w, new_h = self._digital_zoom_crop_rect(w, h)
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # Crop
        cropped = frame[y1:y2, x1:x2]
        
        # Resize back to original size (Digital Zoom effect).
        # CUBIC preserves edges better than LINEAR when enlarging crops.
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return zoomed

    def set_zoom(self, level: float):
        """
        Set the target zoom level.
        Actual processing happens in apply_digital_zoom().
        """
        if level in self.zoom_levels:
            self.current_zoom_level = level
            if level <= 1.0:
                self.reset_roi_center()
            else:
                self._roi_nx, self._roi_ny = self._clamp_roi_norm(self._roi_nx, self._roi_ny, zoom_level=level)
            print(f"Action: Setting Zoom to {level}x")
            if self.logger is not None:
                self.logger.log_event(
                    "zoom_action",
                    zoom=float(level),
                    roi_nx=float(self._roi_nx),
                    roi_ny=float(self._roi_ny),
                )
        else:
            print(f"Warning: Invalid zoom level {level}. Ignoring.")

    def _check_exposure_support(self) -> bool:
        """
        Test if the camera supports exposure control.
        """
        print("Checking exposure support...")
        
        # 1. Try to get current value
        initial_val = self.camera.get_property(cv2.CAP_PROP_EXPOSURE)
        print(f"Initial Exposure: {initial_val}")
        
        # 2. Try to set a different value (e.g., -6)
        test_val = -6.0
        # If initial was -6, try -5 to force a change
        if initial_val == -6.0: test_val = -5.0
            
        self.camera.set_property(cv2.CAP_PROP_EXPOSURE, test_val)
        time.sleep(0.5) # Wait for hardware to react
        
        # 3. Read back
        new_val = self.camera.get_property(cv2.CAP_PROP_EXPOSURE)
        print(f"New Exposure: {new_val}")
        
        # 4. Check if it changed
        # Note: Some cameras return approximate values, so we check range
        if new_val != initial_val:
            print(">> Exposure Control: SUPPORTED ✅")
            return True
        else:
            print(">> Exposure Control: NOT SUPPORTED ❌ (or failed to change)")
            return False

    def execute_exposure(self, level_idx: int):
        """
        Execute an exposure action.
        
        Args:
            level_idx: Index in self.exposure_levels list.
        """
        # Clamp index
        idx = max(0, min(level_idx, len(self.exposure_levels) - 1))
        val = self.exposure_levels[idx]
        
        print(f"Action: Setting Exposure to {val}")
        self.camera.set_property(cv2.CAP_PROP_EXPOSURE, val)
        if self.logger is not None:
            self.logger.log_event(
                "exposure_action",
                exposure_index=idx,
                exposure_value=float(val),
            )


# --- Independent Test ---
def run_policy_demo(camera_id: int = 1):
    from src.camera import Camera
    
    print("Initializing Camera for Policy Test...")
    try:
        camera = Camera(camera_id)
    except Exception as e:
        print(e)
        return

    policy = ActionPolicy(camera)
    
    if policy.exposure_supported:
        print("\nStarting Exposure Sweep Test...")
        print("Watch the video window - it should get brighter/darker.")
        
        try:
            # Sweep through all levels
            for i, level in enumerate(policy.exposure_levels):
                policy.execute_exposure(i)
                
                # Show result for 1 second
                start = time.time()
                while time.time() - start < 1.0:
                    ret, frame = camera.read()
                    if ret:
                        # Test Zoom applying
                        # (Normally loop.py does this, but we sim it here)
                        frame = policy.apply_digital_zoom(frame)
                        
                        cv2.putText(frame, f"Exp: {level}", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        camera.display(frame, "Policy Test")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
                        
        except KeyboardInterrupt:
            pass
    else:
        print("\nSkipping Exposure Test (Not Supported).")

    # --- New Zoom Test ---
    print("\nStarting Digital Zoom Test...")
    print("Watch the video window - it should Zoom In.")
    
    try:
        # Reset exposure to middle
        if policy.exposure_supported:
            policy.execute_exposure(4) # -4
            
        for zoom in policy.zoom_levels:
            policy.set_zoom(zoom)
            
            # Show for 2 seconds
            start = time.time()
            while time.time() - start < 2.0:
                ret, frame = camera.read()
                if ret:
                    # Apply Zoom
                    frame = policy.apply_digital_zoom(frame)
                    
                    cv2.putText(frame, f"Zoom: {zoom}x", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    camera.display(frame, "Policy Test")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
    except KeyboardInterrupt:
        pass

    print("Test Finished.")
    camera.release()


def main():
    run_policy_demo()

if __name__ == "__main__":
    main()
