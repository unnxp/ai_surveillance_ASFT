import cv2
import time
import numpy as np

from cameras.camera import Camera
from ai.detector import PersonDetector
from ai.face_recognition import FaceRecognitionManager
from ai.action_classifier import ActionClassifierManager
from logic.risk_assessment import RiskAssessmentEngine

# ════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════
TARGET_FPS          = 30
FRAME_TIME          = 1.0 / TARGET_FPS
DISPLAY_WIDTH       = 640
YOLO_INPUT_SIZE     = 640
CONF_THRESHOLD      = 0.5
PRE_RESIZE_WIDTH    = 960

# ════════════════════════════════════════════════
#  UTILS
# ════════════════════════════════════════════════
def blank_frame(width=640, height=360, text="NO SIGNAL"):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, text, (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return frame


def resize_keep_ratio(frame, target_width):
    h, w = frame.shape[:2]
    ratio = target_width / w
    return cv2.resize(frame, (target_width, int(h * ratio)))


def pre_resize(frame, max_width=PRE_RESIZE_WIDTH):
    h, w = frame.shape[:2]
    if w > max_width:
        return resize_keep_ratio(frame, max_width)
    return frame


def draw_label(frame, text, color=(0, 255, 0), pos=(20, 40)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 3, cv2.LINE_AA)


def draw_fps(frame, fps, active_tracks=0):
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame, f"Tracks: {active_tracks}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)


# ════════════════════════════════════════════════
#  INIT (EDGE NODE 2 - CAMERA 2 PURE EXPORTER)
# ════════════════════════════════════════════════

detector_cam2 = PersonDetector(
    conf=CONF_THRESHOLD,
    imgsz=YOLO_INPUT_SIZE,
    device=0,
    half=True,
    name="Detector-Cam2",
)

cam2 = Camera(
    r"C:\Users\M S I\Desktop\project_main\ai_surveillance\videos\test_for_cam2.mp4",
    name="Camera 2 (Always ON)",
).start()

face_manager = FaceRecognitionManager()
action_manager = ActionClassifierManager()
risk_engine = RiskAssessmentEngine()

fps_counter = 0
fps_display = 0.0
fps_timer = time.time()

# ════════════════════════════════════════════════
#  MAIN LOOP (CAMERA 2 - EDGE FEATURE EXPORTER)
# ════════════════════════════════════════════════
while True:
    loop_start = time.time()
    now = loop_start

    fps_counter += 1
    if now - fps_timer >= 1.0:
        fps_display = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer = now

    ret2, frame2 = cam2.read()

    if ret2 and frame2 is not None:
        pre2 = pre_resize(frame2)
        scale_cam2 = DISPLAY_WIDTH / pre2.shape[1]

        result2 = detector_cam2.track(pre2)
        view2 = resize_keep_ratio(result2.plot(), DISPLAY_WIDTH)
        
        active_tracks2 = len(result2.boxes.id) if (result2 is not None and result2.boxes is not None and result2.boxes.id is not None) else 0

        h2, w2 = view2.shape[:2]
        risk_states2 = risk_engine.update_and_assess(
            cam2.name, result2, scale=scale_cam2, frame_w=w2, frame_h=h2, current_time=now,
            face_manager=face_manager, action_manager=action_manager, raw_frame=pre2, check_interval=1.0
        )
        
        view2 = risk_engine.heatmap_manager.apply_overlay(cam2.name, view2, risk_engine.is_off_hours())
        
        risk_engine.draw_geofence(view2, cam2.name)
        risk_engine.draw_risk_overlay(view2, result2, scale_cam2, risk_states2)

        draw_label(view2, "ACTIVE", (0, 255, 0))
        draw_fps(view2, fps_display, active_tracks2)
        
        hm_mode = risk_engine.heatmap_manager.get_display_mode_str()
        if hm_mode != "OFF":
            cv2.putText(view2, f"HEATMAP: {hm_mode}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    else:
        view2 = blank_frame(text="CAMERA 2 OFFLINE")

    cv2.imshow(cam2.name, view2)

    elapsed = time.time() - loop_start
    wait_ms = max(1, int((FRAME_TIME - elapsed) * 1000))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        new_mode = risk_engine.heatmap_manager.toggle_mode()
        print(f"[UI] Heatmap mode toggled to: {new_mode}")
    elif key == ord('j'):
        json_packet_str = risk_engine.export_edge_packet(cam2.name, current_time=now, to_json=True, indent=2)
        print("\n==================== [EDGE JSON PACKET - CAM 2] ====================")
        print(json_packet_str)
        print("============================================================\n")

cam2.release()
cv2.destroyAllWindows()
