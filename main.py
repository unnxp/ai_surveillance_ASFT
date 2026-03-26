import cv2
import time
import numpy as np

from cameras.camera import Camera
from ai.detector import PersonDetector
from clients.pir_client import PIRClient
import threading

# ════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════
TARGET_FPS         = 30
FRAME_TIME         = 1.0 / TARGET_FPS
DISPLAY_WIDTH      = 640
YOLO_INPUT_SIZE    = 640        # ลองเป็น 416 ถ้าอยาก FPS สูงขึ้น
SKIP_FRAMES        = 2          # ลองเป็น 3 ถ้า GPU ยัง load สูง
CONF_THRESHOLD     = 0.5
PRE_RESIZE_WIDTH   = 960        # ✅ resize ก่อนส่ง YOLO (ช่วย 4K มาก)
CAM2_HOLD_DURATION = 3.0
PIR_POLL_INTERVAL  = 0.5

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
    """
    ✅ Resize ก่อนส่ง YOLO
    ลด memory bandwidth โดยเฉพาะวิดีโอ 4K
    """
    h, w = frame.shape[:2]
    if w > max_width:
        return resize_keep_ratio(frame, max_width)
    return frame


def draw_label(frame, text, color, pos=(20, 40)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


def draw_fps(frame, fps, pos=(20, 75)):
    cv2.putText(frame, f"FPS: {fps:.1f}", pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# ════════════════════════════════════════════════
#  PIR POLLER (background thread)
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
detector = PersonDetector(
    conf=CONF_THRESHOLD,
    imgsz=YOLO_INPUT_SIZE,
    skip_frames=SKIP_FRAMES,
    device=0,
    half=True,              # ✅ FP16
)

cam1 = Camera(
    "videos/3105196-uhd_3840_2160_30fps.mp4",
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

# ---- cam2 state ----
cam2_open = False
last_person_time = 0.0

# ---- FPS ----
fps_counter = 0
fps_display = 0.0
fps_timer = time.time()

# ════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════
while True:
    loop_start = time.time()
    now = loop_start

    # ── FPS ──────────────────────────
    fps_counter += 1
    if now - fps_timer >= 1.0:
        fps_display = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer = now

    # ── Read frames ──────────────────
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    # ── PIR state ────────────────────
    if pir_poller.is_active:
        cam2_open = True

    # ════════════════════════════════════════
    # ✅ BATCH INFERENCE (2 frames → 1 GPU call)
    # ════════════════════════════════════════
    batch_frames = []
    batch_index = {}     # map: {'cam1': 0, 'cam2': 1}

    if ret1 and frame1 is not None:
        batch_frames.append(pre_resize(frame1))
        batch_index['cam1'] = len(batch_frames) - 1

    if ret2 and frame2 is not None and cam2_open:
        batch_frames.append(pre_resize(frame2))
        batch_index['cam2'] = len(batch_frames) - 1

    # รัน inference ครั้งเดียวสำหรับทุก frame
    batch_results = detector.detect_batch(batch_frames) if batch_frames else []

    # ════════ Camera 1 ════════
    if ret1 and 'cam1' in batch_index:
        result1 = batch_results[batch_index['cam1']]
        view1 = resize_keep_ratio(result1.plot(), DISPLAY_WIDTH)
        draw_label(view1, "ACTIVE", (0, 255, 0))
    else:
        view1 = blank_frame(text="CAMERA 1 OFFLINE")

    draw_fps(view1, fps_display)
    cv2.imshow(cam1.name, view1)

    # ════════ Camera 2 ════════
    if not ret2 or frame2 is None:
        view2 = blank_frame(text="CAMERA 2 OFFLINE")

    elif cam2_open and 'cam2' in batch_index:
        result2 = batch_results[batch_index['cam2']]
        person_detected = detector.has_person(result2)

        if person_detected:
            last_person_time = now

        if now - last_person_time <= CAM2_HOLD_DURATION:
            view2 = resize_keep_ratio(result2.plot(), DISPLAY_WIDTH)
            draw_label(view2, "ACTIVE", (0, 255, 0))
        else:
            cam2_open = False
            view2 = resize_keep_ratio(frame2, DISPLAY_WIDTH)
            draw_label(view2, "STANDBY", (0, 165, 255))
    else:
        view2 = resize_keep_ratio(frame2, DISPLAY_WIDTH)
        draw_label(view2, "STANDBY", (0, 165, 255))

    draw_fps(view2, fps_display)
    cv2.imshow(cam2.name, view2)

    # ════════ EXIT ════════
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
