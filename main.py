import cv2
import time
from cameras.camera import Camera
from ai.detector import PersonDetector
TARGET_FPS = 15
FRAME_TIME = 1.0 / TARGET_FPS
def resize_keep_ratio(frame, target_width=640):
    h, w = frame.shape[:2]
    ratio = target_width / w
    return cv2.resize(frame, (target_width, int(h * ratio)))

# สร้าง detector แค่ครั้งเดียว
detector = PersonDetector(conf=0.5)

# สร้างกล้อง 2 ตัว
cam1 = Camera("videos\\853889-hd_1920_1080_25fps.mp4", name="Camera 1")
cam2 = Camera("videos\\15052703_2560_1440_30fps.mp4", name="Camera 2")

while True:
    start_time = time.time()
    # อ่าน frame จากทั้งสองกล้อง
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 and not ret2:
        break

    # ตรวจจับคน
    result1 = detector.detect(frame1)
    result2 = detector.detect(frame2)

    # วาดผลลัพธ์
    view1 = resize_keep_ratio(result1.plot())
    view2 = resize_keep_ratio(result2.plot())

    # แสดงผล
    cv2.imshow(cam1.name, view1)
    cv2.imshow(cam2.name, view2)

    # ควบคุม FPS
    elapsed = time.time() - start_time
    sleep_time = max(0, FRAME_TIME - elapsed)
    time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()