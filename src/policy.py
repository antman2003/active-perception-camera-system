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
        self.exposure_supported = self._check_exposure_support()
        
        # Define the "Action Space" (Discrete levels of exposure)
        # Note: OpenCV exposure values are usually log2 seconds.
        # -1 = 640ms, -2 = 320ms ... -5 = 40ms, -6 = 20ms, -7 = 10ms
        self.exposure_levels = [-8, -7, -6, -5, -4, -3,-2]
        
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


# --- Independent Test ---
def main():
    from src.camera import Camera
    
    print("Initializing Camera for Policy Test...")
    # Use ID 1 (your external cam)
    try:
        camera = Camera(1)
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
                while time.time() - start < 2.0:
                    ret, frame = camera.read()
                    if ret:
                        cv2.putText(frame, f"Exp: {level}", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        camera.display(frame, "Policy Test")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
                        
        except KeyboardInterrupt:
            pass
    else:
        print("\nSince exposure is not supported, we need a fallback action (Week 2).")
        print("Press any key to exit.")
        camera.read() # clear buffer
        cv2.waitKey(0)

    camera.release()

if __name__ == "__main__":
    main()
