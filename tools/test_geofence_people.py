import os
import sys
import cv2
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ai.detector import PersonDetector
from logic.risk_assessment import RiskAssessmentEngine

def main():
    parser = argparse.ArgumentParser(description="Manual Verification: Person Geofencing & Heatmap Grid")
    parser.add_argument("--video", type=str, default="videos/3318088_hd_1920_1080_25fps.mp4", 
                        help="Video file path or webcam index (default: videos/853889-hd_1920_1080_25fps.mp4)")
    parser.add_argument("--camera-name", type=str, default="Camera 1 (Always ON)", 
                        help="Camera identifier in config (default: Camera 1 (Always ON))")
    args = parser.parse_args()

    # Load engines
    print("Initializing YOLOv8 Detector & Tracker...")
    detector = PersonDetector(tracker_mode="botsort")
    
    print("Initializing Risk Assessment & Heatmap Engine...")
    risk_engine = RiskAssessmentEngine()
    
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
        
    print(f"\nOpening video source: {args.video}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.video}")
        return

    # Tracking speeds and scale variables
    speeds = {}  # Mock speeds
    last_positions = {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    time_step = 1.0 / fps

    # Configure OpenCV scaling window to prevent cropping on smaller screens
    window_name = "Smart Surveillance - Geofence & Heatmap Test (Manual)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    display_w = 960
    display_h = int(vid_h * (display_w / vid_w))
    cv2.resizeWindow(window_name, display_w, display_h)

    last_grid_print = 0.0
    frame_count = 0

    print("\n--- Manual Test Started ---")
    print("Commands:")
    print("  - Press 'q' to quit the test")
    print("  - Press 'h' to toggle heatmap display modes (OFF -> SHORT-TERM -> LONG-TERM)")
    print("Terminal logs will print when a person breaches the Geofence.")
    print("An ASCII representation of the 32x32 Heatmap grid will print in the console every 5 seconds.")
    print("----------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame read failed.")
            break
            
        frame_h, frame_w = frame.shape[:2]
        current_time = frame_count * time_step
        
        # 1. Run YOLOv8 Tracking
        result = detector.track(frame)
        
        # Calculate speeds mock helper
        current_positions = {}
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                xyxy = boxes.xyxy[i].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                current_positions[track_id] = (cx, cy)
                
                if track_id in last_positions:
                    lx, ly = last_positions[track_id]
                    dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                    speeds[track_id] = dist * fps  # pixels per second
                else:
                    speeds[track_id] = 0.0
                    
        last_positions = current_positions
        
        # 2. Update Risk Assessment and Heatmaps
        # scale=1.0 as we are working directly in pixels
        active_states = risk_engine.update_and_assess(
            camera_name=args.camera_name,
            result=result,
            speeds=speeds,
            scale=1.0,
            frame_w=frame_w,
            frame_h=frame_h,
            current_time=current_time
        )
        
        # 3. Draw Geofence Polygons (Yellow)
        polygons = risk_engine.get_geofence_polygons(args.camera_name, frame_w, frame_h)
        for name, poly in polygons:
            cv2.polylines(frame, [poly], True, (0, 255, 255), 2)
            cv2.putText(frame, f"Geofence: {name}", (poly[0][0], poly[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 4. Render Bounding Boxes & Detect Breaches
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id != 0:
                    continue  # Only display person boxes for this test
                    
                track_id = int(boxes.id[i].item())
                xyxy = [int(val) for val in boxes.xyxy[i].tolist()]
                
                state = active_states.get(track_id)
                is_breach = state.is_inside_geofence if state else False
                
                if is_breach:
                    # Bounding Box: Flashing Red/Yellow or Bold Red
                    box_color = (0, 0, 255)
                    label = f"ID: {track_id} [BREACH: {state.geofence_name}]"
                    
                    # Print log directly to Terminal
                    if frame_count % 25 == 0:  # Print roughly once per second in log
                        print(f"Log Alert: Person ID {track_id} breached restricted area '{state.geofence_name}' at time {current_time:.1f}s")
                        
                    # Draw checking points (feet, knee, waist) for transparency
                    p_cx = (xyxy[0] + xyxy[2]) // 2
                    p_cy = (xyxy[1] + xyxy[3]) // 2
                    p_knee_y = (p_cy + xyxy[3]) // 2
                    
                    cv2.circle(frame, (p_cx, xyxy[3]), 5, (255, 255, 0), -1) # Foot
                    cv2.circle(frame, (p_cx, p_knee_y), 5, (255, 255, 0), -1) # Knee
                    cv2.circle(frame, (p_cx, p_cy), 5, (255, 255, 0), -1) # Waist/Center
                else:
                    box_color = (0, 255, 0) # Green for normal
                    label = f"ID: {track_id}"

                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # 5. Apply Heatmap Overlay (Toggleable)
        is_off_hours_active = risk_engine.is_off_hours()
        frame = risk_engine.heatmap_manager.apply_overlay(args.camera_name, frame, is_off_hours_active)

        # Draw current Heatmap display mode in top right corner
        mode_str = risk_engine.heatmap_manager.get_display_mode_str()
        cv2.putText(frame, f"Heatmap Mode: {mode_str} (Press 'H' to toggle)", (frame_w - 380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 6. Periodic Heatmap 32x32 Grid ASCII Print to Console
        real_time_clock = time.time()
        if real_time_clock - last_grid_print >= 5.0:
            grid = risk_engine.heatmap_manager.get_compressed_grid(args.camera_name, is_off_hours_active)
            grid_2d = np.array(grid).reshape(32, 32)
            
            print(f"\n--- 32x32 Heatmap Grid ASCII Preview (Time: {current_time:.1f}s) ---")
            print("  (Legend: '#' = High density, 'o' = Med density, '.' = Low density, ' ' = Empty)")
            print("  +--------------------------------+")
            for r in range(0, 32, 2):  # Downsample to 16x16 representation for cleaner console fit
                line = "  |"
                for c in range(0, 32, 2):
                    val = grid_2d[r, c]
                    if val == 0.0:
                        line += "  "
                    elif val < 0.3:
                        line += ". "
                    elif val < 0.7:
                        line += "o "
                    else:
                        line += "# "
                line += "|"
                print(line)
            print("  +--------------------------------+")
            print("-------------------------------------------------")
            last_grid_print = real_time_clock

        # 7. Render Screen
        cv2.imshow("Smart Surveillance - Geofence & Heatmap Test (Manual)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Manual test terminated by user.")
            break
        elif key == ord('h') or key == ord('H'):
            new_mode = risk_engine.heatmap_manager.toggle_mode()
            print(f"Switched heatmap display mode to: {new_mode}")
            
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished successfully.")

if __name__ == "__main__":
    main()
