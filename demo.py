"""
Active Perception Camera System - Demo Entry Point

This script starts the complete active perception loop (Exposure + Zoom Control).
It handles command-line arguments to make it easy to switch cameras or settings
without modifying the core logic files.
"""

import argparse
import sys
from src.loop import ActivePerceptionLoop

def parse_args():
    parser = argparse.ArgumentParser(description="Active Perception System Demo")
    parser.add_argument(
        "--cam", 
        type=int, 
        default=1, 
        help="Camera device index (0 for built-in laptop cam, 1 for external USB cam. Default: 1)"
    )
    return parser.parse_args()

def print_welcome_message(cam_id):
    print("="*60)
    print("    Active Perception Camera System - Week 2 Demo")
    print("="*60)
    print(f"[*] Trying to connect to Camera {cam_id}...")
    print("[*] Make sure you have a 6x6 ArUco Marker ready.")
    print("[*] Features Active:")
    print("    - Auto-Exposure Sweep (on lighting change)")
    print("    - Auto-Digital Zoom (on target distance change)")
    print("    - Sniper Recovery Mode (if target is lost while zoomed)")
    print("    - Visual Servoing (Smooth ROI Tracking)")
    print("-" * 60)
    print("    Press 'q' in the video window to quit.")
    print("="*60)

def main():
    args = parse_args()
    
    print_welcome_message(args.cam)
    
    try:
        # Initialize the system with the specified camera ID
        app = ActivePerceptionLoop(camera_id=args.cam)
        # Run the infinite perception loop
        app.run()
    except RuntimeError as e:
        print(f"\n[ERROR] Failed to start system: {e}")
        print(f"        Is camera {args.cam} connected and not used by another program?")
        print("        Try running with '--cam 0' for your built-in webcam.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[i] Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
