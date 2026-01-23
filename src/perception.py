"""
Perception module for active perception system.

Handles visual detection tasks (ArUco markers, etc.).
Designed to provide "Perception Confidence" signals to the main loop.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Any

class PerceptionSystem:
    """
    Wrapper for ArUco marker detection.
    
    Why a Class?
    - To load the ArUco dictionary once (it's heavy) and reuse it.
    - To keep detection parameters configurable.
    """
    
    def __init__(self, marker_dict_id=cv2.aruco.DICT_6X6_250):
        """
        Initialize perception resources.
        
        Args:
            marker_dict_id: Which ArUco dictionary to use. 
                          DICT_6X6_250 is a common standard (6x6 bits, 250 IDs).
        """
        # Load the dictionary (the "vocabulary" of markers we can recognize)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict_id)
        
        # Initialize detection parameters (using default settings for now)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Create the detector object (OpenCV 4.7+ style)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        print(f"Perception initialized with dict ID: {marker_dict_id}")

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Any]:
        """
        Detect ArUco markers in a frame.
        
        Args:
            frame: Input image (BGR) from camera.
            
        Returns:
            (detected, ids, corners):
                - detected (bool): True if ANY marker is found.
                - ids (np.ndarray or None): List of ALL found IDs.
                - corners (list): List of corners for visualization.
        """
        if frame is None:
            return False, None, None

        # Convert to grayscale (detection works better/faster on gray images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Core detection step
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            return True, ids, corners
        else:
            return False, None, None

    def visualize(self, frame: np.ndarray, corners: Any, ids: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw bounding boxes and IDs on the frame.
        """
        if corners is None or ids is None:
            return frame
            
        # Create a copy so we don't modify the original frame
        annotated_frame = frame.copy()
        
        # 1. Draw Total Count at Top-Left
        total_count = len(ids)
        cv2.putText(annotated_frame, f"Total Markers: {total_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 2. Draw each marker
        for i, corner_set in enumerate(corners):
            # Get the ID for this specific marker
            this_id = ids[i][0]
            
            # Convert to integer points
            points = corner_set[0].astype(np.int32)
            
            # Draw the box (Green, thickness=4)
            cv2.polylines(annotated_frame, [points], isClosed=True, color=(0, 255, 0), thickness=4)
            
            # Draw the corner point (Red dot)
            cv2.circle(annotated_frame, tuple(points[0]), 5, (0, 0, 255), -1)
            
            # Draw ID above the box (Top-Left corner)
            text_pos = tuple(points[0])
            text_pos = (text_pos[0], text_pos[1] - 10) # Move up slightly
            
            cv2.putText(annotated_frame, f"ID:{this_id}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 0), thickness=2)

        return annotated_frame


# --- Test Code (Independent Run) ---
def main():
    """
    Test perception module using the camera.
    """
    from src.camera import Camera
    
    print("Initializing Camera & Perception...")
    try:
        # 1. Setup
        camera = Camera(1)  # Use ID 1 (external) based on your setup
        perception = PerceptionSystem()
        
        print("System Ready. Show an ArUco marker to the camera!")
        print("Press 'q' in the window to quit.")
        
        while True:
            # 2. Loop: Sense -> Perceive -> Visualize
            ret, frame = camera.read()
            if not ret:
                break
                
            # Run detection
            detected, ids, corners = perception.detect(frame)
            
            # Visualize result
            # Pass the full list of IDs to visualize
            vis_frame = perception.visualize(frame, corners, ids)
            
            # Print status to terminal (simple logging)
            if detected:
                # We show the Main ID (first one) and the total count
                main_id = ids[0][0]
                count = len(ids)
                print(f"\rMain ID: {main_id} | Total: {count}   ", end="")
            else:
                print(f"\rSearching...         ", end="")
                
            camera.display(vis_frame, "Perception Test")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            camera.release()
        except:
            pass
        print("\nExited.")

if __name__ == "__main__":
    main()
