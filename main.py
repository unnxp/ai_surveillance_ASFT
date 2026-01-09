import cv2
from ultralytics import YOLO

# 1) โหลดโมเดล
model = YOLO("yolov8n.pt")

# 2) เปิดวิดีโอ (แทนกล้อง)
cap = cv2.VideoCapture("videos\\3105196-uhd_3840_2160_30fps.mp4")

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

# 3) loop อ่าน frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    # 4) ส่ง frame เข้า YOLO
    results = model(frame, conf=0.5, classes=[0,24,28,43],device=0)  # class 0 = person , 24 = backpack, 28 = suitcase , 43 = knife

    # 5) วาด bounding box
    annotated_frame = results[0].plot()

    # 6) แสดงผล
    cv2.imshow("Camera 1 - Person Detection", annotated_frame)

    # กด q เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()