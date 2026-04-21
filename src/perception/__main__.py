"""Run: ``python -m src.perception`` — ArUco camera smoke test."""

from src.perception.aruco import run_aruco_camera_demo

if __name__ == "__main__":
    run_aruco_camera_demo(camera_id=1)
