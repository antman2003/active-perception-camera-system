"""
Camera module for active perception system.

Handles camera capture, display, and property setting (exposure, brightness, etc.).
Designed to be simple, reliable, and extensible for closed-loop perception.

Core Responsibilities:
1. Sensing (read): Capture visual data from the environment.
2. Action (set_property): Modify physical imaging parameters (Active Perception).
"""

import cv2
import time
from typing import Tuple, Optional


class Camera:
    """
    Camera wrapper for OpenCV VideoCapture.
    
    Why a Class?
    - State Management: Holds the connection object (self.cap) persistently.
    - Unified Interface: Decouples logic from low-level OpenCV calls.
    - Extensibility: Easy to swap out 'read' logic later (e.g., for different hardware).
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera hardware connection.
        
        Args:
            camera_id: Device index (0 is usually default USB/Integrated cam).
            
        Raises:
            RuntimeError: If hardware cannot be accessed (Fail Fast).
        """
        # Connect to hardware
        # On Windows, cv2.CAP_DSHOW makes startup much faster (DirectShow)
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        # Critical Check: Validate hardware connection immediately.
        # Don't wait until runtime read() calls to fail.
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Store ID for reference/logging
        self.camera_id = camera_id
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a single frame (The 'Sensing' part).
        
        Returns:
            (success, frame): 
                - success: False if frame is dropped/camera disconnected.
                - frame: The visual data (numpy array) or None.
        """
        return self.cap.read()
    
    def display(self, frame, window_name: str = "Camera") -> None:
        """
        Helper for visualization/debugging.
        """
        cv2.imshow(window_name, frame)
    
    def set_property(self, prop: int, value: float) -> bool:
        """
        Set a camera property (The 'Action' part).
        
        This is the KEY interface for Active Perception.
        It allows the system to physically change how it sees the world
        (e.g., changing exposure when image is too dark).
        
        Args:
            prop: OpenCV property constant (e.g., cv2.CAP_PROP_EXPOSURE)
            value: The target value.
            
        Returns:
            True if the driver accepted the command.
        """
        return self.cap.set(prop, value)
    
    def get_property(self, prop: int) -> float:
        """Read current camera settings."""
        return self.cap.get(prop)
    
    def is_opened(self) -> bool:
        """Check connection status."""
        return self.cap.isOpened()
    
    def release(self) -> None:
        """
        Graceful cleanup.
        Release hardware lock so other apps can use the camera.
        """
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Independent Test Entry Point.
    Run this file directly to verify hardware: 'python -m src.camera'
    """
    # TIP: Change this ID if you have multiple cameras.
    # 0 = Integrated Webcam (usually)
    # 1 = External USB Camera (usually)
    target_camera_id = 1
    
    print(f"Opening camera {target_camera_id}...")
    try:
        camera = Camera(target_camera_id)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Try changing 'target_camera_id' in main() to 0 or another number.")
        return
    
    print("Camera opened. Press 'q' IN THE VIDEO WINDOW to quit.")
    
    # FPS Measurement (First 2 seconds only)
    start_time = time.time()
    frame_count = 0
    fps_measured = False
    
    try:
        while True:
            # 1. Sense
            ret, frame = camera.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # 2. Visualize
            camera.display(frame)
            
            # --- Measure FPS (One-time in first 2s) ---
            if not fps_measured:
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 2.0:
                    fps = frame_count / elapsed
                    print(f"Measured Camera FPS: {fps:.2f}")
                    fps_measured = True
            # ------------------------------

            # Check for 'q' key press
            # IMPORTANT: Focus must be on the video window, not the terminal!
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            
            # Allow quitting by closing the window with the mouse (X button)
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Always clean up resources, even on error
        camera.release()
        print("Camera released.")


if __name__ == "__main__":
    main()
