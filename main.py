import cv2
import time
import threading
import numpy as np

from cameras.camera import Camera
from ai.detector import PersonDetector
from ai.tracker import TrajectoryManager
from clients.pir_client import PIRClient
from logic.risk_assessment import RiskAssessmentEngine
from logic.alert_manager import AlertManager

# ════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════
TARGET_FPS          = 30
FRAME_TIME          = 1.0 / TARGET_FPS
DISPLAY_WIDTH       = 640
YOLO_INPUT_SIZE     = 640
CONF_THRESHOLD      = 0.5
PRE_RESIZE_WIDTH    = 960
CAM2_HOLD_DURATION  = 3.0
PIR_POLL_INTERVAL   = 0.5
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
#  PIR POLLER
# ════════════════════════════════════════════════
class PIRPoller:
    def __init__(self, client: PIRClient, interval: float = 0.5):
        self._client = client
        self._interval = interval
        self._active = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="Thread-PIR"
        )

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        while True:
            try:
                state = self._client.is_active()
            except Exception:
                state = False
            with self._lock:
                self._active = state
            time.sleep(self._interval)

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active


# ════════════════════════════════════════════════
#  INIT
# ════════════════════════════════════════════════

# ✅ แยก detector ต่อกล้อง → tracker state แยกกัน 100%
detector_cam1 = PersonDetector(
    conf=CONF_THRESHOLD,
    imgsz=YOLO_INPUT_SIZE,
    device=0,
    half=True,
    name="Detector-Cam1",
)
detector_cam2 = PersonDetector(
    conf=CONF_THRESHOLD,
    imgsz=YOLO_INPUT_SIZE,
    device=0,
    half=True,
    name="Detector-Cam2",
)

traj_cam1 = TrajectoryManager(
    max_history=TRAJ_MAX_HISTORY,
    max_age=TRAJ_MAX_AGE,
    predict_steps=PREDICT_STEPS,
)
traj_cam2 = TrajectoryManager(
    max_history=TRAJ_MAX_HISTORY,
    max_age=TRAJ_MAX_AGE,
    predict_steps=PREDICT_STEPS,
)

cam1 = Camera(
    # "videos/3105196-uhd_3840_2160_30fps.mp4", # ✅ เก็บไว้สำหรับเปลี่ยนกลับมาทดสอบ
    "rtsp://admin:password@192.168.1.100:554/stream1",
    name="Camera 1 (Always ON)"
).start()

cam2 = Camera(
    r"C:\Users\M S I\Desktop\project_main\ai_surveillance\videos\test_for_cam2.mp4",
    name="Camera 2 (PIR + Self Hold)",
).start()

pir_poller = PIRPoller(
    PIRClient("http://127.0.0.1:8000"),
    interval=PIR_POLL_INTERVAL
).start()

# ✅ เริ่มต้นระบบวิเคราะห์ความเสี่ยงและการแจ้งเตือน
risk_engine = RiskAssessmentEngine()
alert_manager = AlertManager(risk_engine.config)

cam2_open = False
last_person_time = 0.0
fps_counter = 0
fps_display = 0.0
fps_timer = time.time()

# ════════════════════════════════════════════════
#  MAIN LOOP
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

    # ── Read frames ──────────────────────────────
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    # ── PIR ──────────────────────────────────────
    if pir_poller.is_active:
        cam2_open = True

    # ════════ Camera 1 (Always ON) ════════════════
    if ret1 and frame1 is not None:
        pre1 = pre_resize(frame1)
        scale_cam1 = DISPLAY_WIDTH / pre1.shape[1]

        # ✅ detector_cam1 มี tracker state ของตัวเอง
        result1 = detector_cam1.track(pre1)

        speeds1 = traj_cam1.update(result1, scale=scale_cam1)
        view1 = resize_keep_ratio(result1.plot(), DISPLAY_WIDTH)
        traj_cam1.draw(view1, result1, scale=scale_cam1)
        
        # ✅ วิเคราะห์ความเสี่ยงและ Geofence
        h1, w1 = view1.shape[:2]
        risk_states1 = risk_engine.update_and_assess(
            cam1.name, result1, speeds1, scale_cam1, w1, h1, now
        )
        risk_engine.draw_geofence(view1, cam1.name)
        risk_engine.draw_risk_overlay(view1, result1, scale_cam1, risk_states1)
        
        # ✅ ทริกเกอร์แจ้งเตือน/บันทึกผล
        alert_manager.check_and_trigger_alert(cam1.name, view1, risk_states1, now)
        
        # ✅ แสดงผลแจ้งเตือนกระพริบหากมีความเสี่ยงระดับสูง
        has_alarm1 = any(s.current_level in ["HIGH", "CRITICAL"] for s in risk_states1.values())
        if has_alarm1:
            border_color = (0, 0, 255) if (int(now * 2) % 2 == 0) else (0, 165, 255)
            cv2.rectangle(view1, (0, 0), (w1 - 1, h1 - 1), border_color, 6)
            cv2.putText(view1, "CRITICAL ALARM DETECTED", (w1 // 2 - 130, h1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)

        draw_label(view1, "ACTIVE", (0, 255, 0))
        draw_fps(view1, fps_display, traj_cam1.active_count())
    else:
        view1 = blank_frame(text="CAMERA 1 OFFLINE")

    cv2.imshow(cam1.name, view1)

    # ════════ Camera 2 (PIR-triggered) ════════════
    if not ret2 or frame2 is None:
        view2 = blank_frame(text="CAMERA 2 OFFLINE")

    elif cam2_open:
        pre2 = pre_resize(frame2)
        scale_cam2 = DISPLAY_WIDTH / pre2.shape[1]

        # ✅ detector_cam2 มี tracker state ของตัวเอง
        result2 = detector_cam2.track(pre2)

        person_detected = detector_cam2.has_person(result2)
        if person_detected:
            last_person_time = now

        if now - last_person_time <= CAM2_HOLD_DURATION:
            speeds2 = traj_cam2.update(result2, scale=scale_cam2)
            view2 = resize_keep_ratio(result2.plot(), DISPLAY_WIDTH)
            traj_cam2.draw(view2, result2, scale=scale_cam2)
            
            # ✅ วิเคราะห์ความเสี่ยงและ Geofence
            h2, w2 = view2.shape[:2]
            risk_states2 = risk_engine.update_and_assess(
                cam2.name, result2, speeds2, scale_cam2, w2, h2, now
            )
            risk_engine.draw_geofence(view2, cam2.name)
            risk_engine.draw_risk_overlay(view2, result2, scale_cam2, risk_states2)
            
            # ✅ ทริกเกอร์แจ้งเตือน/บันทึกผล
            alert_manager.check_and_trigger_alert(cam2.name, view2, risk_states2, now)
            
            # ✅ แสดงผลแจ้งเตือนกระพริบหากมีความเสี่ยงระดับสูง
            has_alarm2 = any(s.current_level in ["HIGH", "CRITICAL"] for s in risk_states2.values())
            if has_alarm2:
                border_color = (0, 0, 255) if (int(now * 2) % 2 == 0) else (0, 165, 255)
                cv2.rectangle(view2, (0, 0), (w2 - 1, h2 - 1), border_color, 6)
                cv2.putText(view2, "CRITICAL ALARM DETECTED", (w2 // 2 - 130, h2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)

            draw_label(view2, "ACTIVE", (0, 255, 0))
            draw_fps(view2, fps_display, traj_cam2.active_count())
        else:
            cam2_open = False
            traj_cam2.reset()
            view2 = resize_keep_ratio(frame2, DISPLAY_WIDTH)
            # วาด geofence บน standby feed
            risk_engine.draw_geofence(view2, cam2.name)
            draw_label(view2, "STANDBY", (0, 165, 255))
            draw_fps(view2, fps_display)
    else:
        view2 = resize_keep_ratio(frame2, DISPLAY_WIDTH)
        # วาด geofence บน standby feed
        risk_engine.draw_geofence(view2, cam2.name)
        draw_label(view2, "STANDBY", (0, 165, 255))
        draw_fps(view2, fps_display)

    cv2.imshow(cam2.name, view2)

    # ════════ EXIT ════════════════════════════════
    elapsed = time.time() - loop_start
    wait_ms = max(1, int((FRAME_TIME - elapsed) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

# ════════════════════════════════════════════════
#  CLEANUP
# ════════════════════════════════════════════════
cam1.release()
cam2.release()
cv2.destroyAllWindows()
