"""ArUco marker detection (original Session 1–20 perception)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from src.perception.base import PerceptionDetector


class ArucoDetector(PerceptionDetector):
    """Loads dictionary once; `detect` / `visualize` match legacy `PerceptionSystem`."""

    def __init__(self, marker_dict_id=cv2.aruco.DICT_6X6_250):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        print(f"Perception (ArUco) initialized with dict ID: {marker_dict_id}")

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Any]:
        if frame is None:
            return False, None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            return True, ids, corners
        return False, None, None

    def visualize(
        self, frame: np.ndarray, corners: Any, ids: Optional[np.ndarray]
    ) -> np.ndarray:
        if corners is None or ids is None:
            return frame
        annotated_frame = frame.copy()
        total_count = len(ids)
        cv2.putText(
            annotated_frame,
            f"Total Markers: {total_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        for i, corner_set in enumerate(corners):
            this_id = ids[i][0]
            points = corner_set[0].astype(np.int32)
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=4,
            )
            cv2.circle(annotated_frame, tuple(points[0]), 5, (0, 0, 255), -1)
            text_pos = tuple(points[0])
            text_pos = (text_pos[0], text_pos[1] - 10)
            cv2.putText(
                annotated_frame,
                f"ID:{this_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2,
            )
        return annotated_frame


def run_aruco_camera_demo(camera_id: int = 1) -> None:
    from src.camera import Camera

    print("Initializing Camera & ArucoDetector...")
    try:
        camera = Camera(camera_id)
        perception = ArucoDetector()
        print("System Ready. Show an ArUco marker to the camera!")
        print("Press 'q' in the window to quit.")
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            detected, ids, corners = perception.detect(frame)
            vis_frame = perception.visualize(frame, corners, ids)
            if detected:
                main_id = ids[0][0]
                count = len(ids)
                print(f"\rMain ID: {main_id} | Total: {count}   ", end="")
            else:
                print(f"\rSearching...         ", end="")
            camera.display(vis_frame, "Perception Test")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            camera.release()
        except Exception:
            pass
        print("\nExited.")
