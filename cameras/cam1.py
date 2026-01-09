import cv2

def get_frame():
    """
    Generator สำหรับกล้องตัวที่ 1
    ตอนนี้อ่านจากไฟล์ video
    อนาคตสามารถสลับไปใช้ IP Camera ได้
    """

    # ===============================
    # OPTION 1: ใช้ไฟล์วิดีโอ (ตอนนี้)
    # ===============================
    video_path = "C://Users//M S I//Desktop//project_5_Stars//3318088_hd_1920_1080_25fps.mp4"
    cap = cv2.VideoCapture(video_path)

    # ===============================
    # OPTION 2: ใช้ IP Camera (ยังไม่เปิดใช้)
    # ===============================
    # ip_cam_url = "rtsp://user:password@ip_address:554/stream"
    # cap = cv2.VideoCapture(ip_cam_url)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 1 source")

    while True:
        ret, frame = cap.read()

        # วิดีโอจบ → วนใหม่ (เหมาะกับ demo / dev)
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        yield frame

    cap.release()
