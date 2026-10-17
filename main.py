import cv2
import time
import threading
import numpy as np

from cameras.camera import Camera
from ai.detector import PersonDetector
from ai.tracker import TrajectoryManager
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
CONF_THRESHOLD      = 0.35
PRE_RESIZE_WIDTH    = 960
TRAJ_MAX_HISTORY    = 60
TRAJ_MAX_AGE        = 5.0
PREDICT_STEPS       = 20

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
#  INIT (EDGE NODE 1 - CAMERA 1 PURE EXPORTER)
# ════════════════════════════════════════════════

detector_cam1 = PersonDetector(
    conf=CONF_THRESHOLD,
    imgsz=YOLO_INPUT_SIZE,
    device=0,
    half=True,
    name="Detector-Cam1",
)

cam1 = Camera(
    "videos/3105196-uhd_3840_2160_30fps.mp4", # ✅ เก็บไว้สำหรับเปลี่ยนกลับมาทดสอบ
    # "rtsp://admin:<PASSWORD>@192.168.137.17:554/stream1",
    name="Camera 1 (Always ON)"
).start()

# ✅ เริ่มต้นระบบการประมวลผลสกัดฟีเจอร์และระบุตัวตนฝั่ง Edge Node
face_manager = FaceRecognitionManager()
action_manager = ActionClassifierManager()
risk_engine = RiskAssessmentEngine()

fps_counter = 0
fps_display = 0.0
fps_timer = time.time()

# ════════════════════════════════════════════════
#  MAIN LOOP (CAMERA 1 - EDGE FEATURE EXPORTER)
# ════════════════════════════════════════════════
while True:
    loop_start = time.time()
    now = loop_start

    # ── FPS ──────────────────────────────────────
    fps_counter += 1
    if now - fps_timer >= 1.0:
        fps_display = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer = now

    # ── Read frame ──────────────────────────────
    ret1, frame1 = cam1.read()

    # ════════ Camera 1 (Always ON) ════════════════
    if ret1 and frame1 is not None:
        pre1 = pre_resize(frame1)
        scale_cam1 = DISPLAY_WIDTH / pre1.shape[1]

        # ✅ detector_cam1 มี tracker state ของตัวเอง
        result1 = detector_cam1.track(pre1)
        view1 = resize_keep_ratio(result1.plot(), DISPLAY_WIDTH)
        
        active_tracks1 = len(result1.boxes.id) if (result1 is not None and result1.boxes is not None and result1.boxes.id is not None) else 0

        # ✅ สกัดฟีเจอร์ Geofence, ใบหน้า และจำแนกพฤติกรรม
        h1, w1 = view1.shape[:2]
        risk_states1 = risk_engine.update_and_assess(
            cam1.name, result1, scale=scale_cam1, frame_w=w1, frame_h=h1, current_time=now,
            face_manager=face_manager, action_manager=action_manager, raw_frame=pre1, check_interval=1.0
        )
        
        # ✅ แสดงผลแผนภาพความร้อนสะสมและ Geofence
        view1 = risk_engine.heatmap_manager.apply_overlay(cam1.name, view1, risk_engine.is_off_hours())
        
        risk_engine.draw_geofence(view1, cam1.name)
        risk_engine.draw_risk_overlay(view1, result1, scale_cam1, risk_states1)

        draw_label(view1, "ACTIVE", (0, 255, 0))
        draw_fps(view1, fps_display, active_tracks1)
        
        # แสดงสถานะโหมด Heatmap
        hm_mode = risk_engine.heatmap_manager.get_display_mode_str()
        if hm_mode != "OFF":
            cv2.putText(view1, f"HEATMAP: {hm_mode}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    else:
        view1 = blank_frame(text="CAMERA 1 OFFLINE")

    cv2.imshow(cam1.name, view1)

    # ════════ EXIT / KEYBOARD EVENTS ══════════════
    elapsed = time.time() - loop_start
    wait_ms = max(1, int((FRAME_TIME - elapsed) * 1000))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        new_mode = risk_engine.heatmap_manager.toggle_mode()
        print(f"[UI] Heatmap mode toggled to: {new_mode}")
    elif key == ord('j'):
        json_packet_str = risk_engine.export_edge_packet(cam1.name, current_time=now, to_json=True, indent=2)
        print("\n==================== [EDGE JSON PACKET - CAM 1] ====================")
        print(json_packet_str)
        print("============================================================\n")


# ════════════════════════════════════════════════
#  CLEANUP
# ════════════════════════════════════════════════
cam1.release()
cv2.destroyAllWindows()
