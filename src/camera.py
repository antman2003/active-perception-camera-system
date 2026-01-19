"""
Camera module for active perception system.

Handles camera capture, display, and property setting (exposure, brightness, etc.).
Designed to be simple, reliable, and extensible for closed-loop perception.
"""

import cv2
from typing import Tuple, Optional


class Camera:
    """
    Camera wrapper for OpenCV VideoCapture.
    
    Provides a clean interface for:
    - Frame capture
    - Display (optional, for debugging/demo)
    - Property setting (exposure, brightness, etc.) for active perception actions
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera.
        
        Args:
            camera_id: Camera device index (default: 0 for first camera)
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Store camera ID for reference
        self.camera_id = camera_id
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the camera.
        
        Returns:
            (success, frame): Tuple of (bool, numpy array or None)
        """
        return self.cap.read()
    
    def display(self, frame, window_name: str = "Camera") -> None:
        """
        Display a frame in a window.
        
        Args:
            frame: Frame to display
            window_name: Name of the window
        """
        cv2.imshow(window_name, frame)
    
    def set_property(self, prop: int, value: float) -> bool:
        """
        Set a camera property (exposure, brightness, etc.).
        
        Args:
            prop: OpenCV property constant (e.g., cv2.CAP_PROP_EXPOSURE)
            value: Property value to set
            
        Returns:
            True if successful, False otherwise
        """
        return self.cap.set(prop, value)
    
    def get_property(self, prop: int) -> float:
        """
        Get a camera property value.
        
        Args:
            prop: OpenCV property constant
            
        Returns:
            Property value
        """
        return self.cap.get(prop)
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap.isOpened()
    
    def release(self) -> None:
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Simple demo: open camera and display live feed.
    Press 'q' to quit.
    """
    print("Opening camera...")
    camera = Camera(0)
    
    print("Camera opened. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            # Display frame
            camera.display(frame)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera.release()
        print("Camera released.")


if __name__ == "__main__":
    main()
