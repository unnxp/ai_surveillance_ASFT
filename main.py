import cv2
import time
import numpy as np

from cameras.camera import Camera
from ai.detector import PersonDetector
from clients.pir_client import PIRClient

# ---------------- CONFIG ----------------
TARGET_FPS = 15
FRAME_TIME = 1.0 / TARGET_FPS
WIDTH = 640

CAM2_HOLD_DURATION = 3.0   # วินาที: ไม่เจอคนเกินนี้ → ปิด cam2

# ---------------- UTILS ----------------
def blank_frame(width=640, height=360, text="NO SIGNAL"):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame, text,
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )
    return frame


def resize_keep_ratio(frame, target_width=640):
    h, w = frame.shape[:2]
    ratio = target_width / w
    return cv2.resize(frame, (target_width, int(h * ratio)))

# ---------------- INIT ----------------
detector = PersonDetector(conf=0.5)

cam1 = Camera(
    "videos/15052703_2560_1440_30fps.mp4",
    name="Camera 1 (Always ON)"
)

cam2 = Camera(
    "videos/Test_2.mp4",
    name="Camera 2 (PIR + Self Hold)"
)

pir = PIRClient("http://127.0.0.1:8000")

# ---- cam2 internal state ----
cam2_open = False
last_person_time = 0.0

# ---------------- LOOP ----------------
while True:
    start_time = time.time()
    now = start_time

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    # ================= Camera 1 =================
    if ret1:
        result1 = detector.detect(frame1)
        view1 = resize_keep_ratio(result1.plot(), WIDTH)
    else:
        view1 = blank_frame(text="CAMERA 1 OFFLINE")

    cv2.imshow(cam1.name, view1)

    # ================= PIR =================
    pir_active = pir.is_active()

    # PIR เปิด cam2 ครั้งแรก
    if pir_active:
        cam2_open = True

    # ================= Camera 2 =================
    if not ret2:
        view2 = blank_frame(text="CAMERA 2 OFFLINE")

    elif cam2_open:
        # detect เฉพาะตอน cam2 เปิด
        result2 = detector.detect(frame2)
        boxes2 = result2.boxes
        person_detected = boxes2 is not None and len(boxes2) > 0

        if person_detected:
            last_person_time = now  # ต่ออายุการเปิด

        # ตัดสินใจว่าจะเปิดต่อหรือปิด
        if now - last_person_time <= CAM2_HOLD_DURATION:
            view2 = resize_keep_ratio(result2.plot(), WIDTH)
            cv2.putText(
                view2, "ACTIVE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )
        else:
            cam2_open = False
            view2 = resize_keep_ratio(frame2, WIDTH)
            cv2.putText(
                view2, "STANDBY",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )
    else:
        # cam2 ยังไม่ถูกเปิดจาก PIR
        view2 = resize_keep_ratio(frame2, WIDTH)
        cv2.putText(
            view2, "STANDBY",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    cv2.imshow(cam2.name, view2)

    # ================= EXIT =================
    if not ret1 and not ret2:
        break

    elapsed = time.time() - start_time
    time.sleep(max(0, FRAME_TIME - elapsed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cam1.release()
cam2.release()
cv2.destroyAllWindows()
