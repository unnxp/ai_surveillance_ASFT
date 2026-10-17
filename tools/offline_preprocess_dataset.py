import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library is not installed. Please install it by running:")
    print("  .\\env\\Scripts\\pip.exe install ultralytics")
    sys.exit(1)

def interpolate_bboxes(bboxes):
    """
    เติมเฟรมที่ขาดหายชั่วคราว (Occlusion/Disappearance) ด้วย Linear Interpolation
    bboxes: dict mapping frame_idx -> [x1, y1, x2, y2]
    """
    if not bboxes:
        return bboxes
        
    frame_indices = sorted(bboxes.keys())
    first_frame = frame_indices[0]
    last_frame = frame_indices[-1]
    
    # วิ่งไล่เฟรมเพื่อทำ Interpolation
    for f in range(first_frame + 1, last_frame):
        if f not in bboxes:
            # หาเฟรมก่อนหน้าที่มีข้อมูล
            prev_f = max([idx for idx in frame_indices if idx < f])
            # หาเฟรมถัดไปที่มีข้อมูล
            next_f = min([idx for idx in frame_indices if idx > f])
            
            # คำนวณอัตราส่วนการแบ่งสัดส่วน
            weight = (f - prev_f) / (next_f - prev_f)
            
            box_prev = np.array(bboxes[prev_f])
            box_next = np.array(bboxes[next_f])
            
            # ทำ Linear Interpolation ของกล่อง Bounding Box
            box_interp = (1 - weight) * box_prev + weight * box_next
            bboxes[f] = box_interp.tolist()
            
    return bboxes

def letterbox_image(image, target_size=(224, 224)):
    """
    ปรับขนาดรูปภาพโดยรักษาสัดส่วนความกว้างต่อความสูงเดิม (Aspect Ratio)
    และเติมขอบดำ (Zero Padding) เพื่อไม่ให้ภาพบีบอัดบิดเบี้ยว
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # คำนวณสเกลย่อขยายเพื่อไม่ให้สัดส่วนบิดเบี้ยว
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # ย่อขยายภาพโดยรักษาสัดส่วน
    resized = cv2.resize(image, (new_w, new_h))

    # สร้างผืนผ้าใบแคนวาสสีดำล้วนขนาด target_size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # วางภาพกึ่งกลางแคนวาสสีดำ
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    return canvas

def preprocess_video(video_path, output_path, model, target_size=(224, 224)):
    """
    รัน YOLOv8 tracking, เลือก ID บุคคลเป้าหมายหลัก, ทำ Interpolation, Crop และเซฟผลลัพธ์
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    frames_cache = []
    frame_idx = 0
    raw_detections = {} # id -> frame_idx -> bbox
    
    # 1. รันอ่านเฟรมทั้งหมดและรัน YOLOv8 Tracking
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frames_cache.append(frame)
        
        # รัน YOLOv8 tracking (ใช้ ByteTrack ทำงานแบบเบื้องหลังพร้อม Kalman Filter)
        results = model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")
        
        # เก็บข้อมูลกรอบคนแต่ละคนแยกตาม ID
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, ids):
                if track_id not in raw_detections:
                    raw_detections[track_id] = {}
                raw_detections[track_id][frame_idx] = box.tolist()
                
        frame_idx += 1
    cap.release()
    
    if len(frames_cache) == 0:
        return False
        
    # 2. ค้นหา ID ที่ปรากฏตัวในวิดีโอมากที่สุด (Primary Target ID)
    primary_id = None
    max_appearances = 0
    for track_id, appearances in raw_detections.items():
        if len(appearances) > max_appearances:
            max_appearances = len(appearances)
            primary_id = track_id
            
    # 3. เตรียม Bounding Boxes ของคนเป้าหมาย
    target_bboxes = {}
    if primary_id is not None:
        target_bboxes = raw_detections[primary_id]
        # ทำ Kalman Interpolation เพื่อกู้เฟรมช่วงที่เป้าหมายเดินหลบหลังเสา
        target_bboxes = interpolate_bboxes(target_bboxes)
        
    # 4. เขียนวิดีโอครอปเฉพาะกรอบตัวบุคคลเก็บไว้ในดิสก์
    height, width = target_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    last_known_crop = None
    
    for f in range(len(frames_cache)):
        frame = frames_cache[f]
        img_h, img_w, _ = frame.shape
        
        if f in target_bboxes:
            # ครอปตามพิกัด Bounding Box พร้อมเพิ่ม margin 50% เพื่อให้เห็นบริบทรอบตัวคน (Context-Aware)
            x1, y1, x2, y2 = map(int, target_bboxes[f])
            w = x2 - x1
            h = y2 - y1
            margin_x = int(w * 0.50)
            margin_y = int(h * 0.50)
            
            # ขยายกรอบ Bounding Box ออกไปด้านละ 50% และป้องกันขอบภาพหลุด
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(img_w, x2 + margin_x)
            y2 = min(img_h, y2 + margin_y)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                last_known_crop = crop
            else:
                crop = last_known_crop if last_known_crop is not None else frame
        else:
            # กรณีไม่มีข้อมูลคนในวิดีโอเลย ให้ดึงเฟรมก่อนหน้าค้างไว้ (Freeze frame)
            crop = last_known_crop if last_known_crop is not None else frame
            
        # ทำ Letterboxing รักษาสัดส่วนพร้อมขอบดำกึ่งกลาง
        crop_letterboxed = letterbox_image(crop, target_size)
        out.write(crop_letterboxed)
        
    out.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Offline Preprocess Video Dataset (Person Bounding Box Crop with Interpolation)")
    parser.add_argument("--dataset-dir", type=str, 
                        default=r"C:\Users\M S I\.cache\kagglehub\datasets\daudshah\video-dataset\versions\1\dataset-video-split",
                        help="Path to the original video dataset split")
    parser.add_argument("--output-dir", type=str, 
                        default=r"C:\Users\M S I\.cache\kagglehub\datasets\daudshah\video-dataset\versions\1\dataset-video-split-preprocessed",
                        help="Path where preprocessed video clips will be saved")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLOv8 model weights for tracking")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing preprocessed files")
    args = parser.parse_args()

    print("==================================================")
    print("  Offline Preprocessing Video Dataset (YOLO + ByteTrack) ")
    print("==================================================")
    print(f"Source Directory: {args.dataset_dir}")
    print(f"Target Directory: {args.output_dir}")

    # โหลด YOLOv8
    print("\nLoading YOLOv8 tracking model...")
    model = YOLO(args.yolo_weights)
    
    splits = ["train", "valid", "test"]
    txt_files = ["train.txt", "valid.txt", "test.txt"]
    
    for split, txt_name in zip(splits, txt_files):
        txt_path = os.path.join(project_root, "exampleDataset", txt_name)
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found. Skipping split: {split}")
            continue
            
        src_split_dir = os.path.join(args.dataset_dir, split)
        dest_split_dir = os.path.join(args.output_dir, split)
        os.makedirs(dest_split_dir, exist_ok=True)
        
        # อ่านรายการไฟล์วิดีโอ
        video_files = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_files.append(parts[0])
                    
        print(f"\nProcessing {split} set ({len(video_files)} videos)...")
        
        for video_name in tqdm(video_files):
            src_video = os.path.join(src_split_dir, video_name)
            dest_video = os.path.join(dest_split_dir, video_name)
            
            if not os.path.exists(src_video):
                continue
                
            if os.path.exists(dest_video) and not args.force:
                # ข้ามหากทำความสะอาดไปแล้ว และไม่ได้เปิดสวิตช์เขียนทับ
                continue
                
            preprocess_video(src_video, dest_video, model)

    print("\n==================================================")
    print("  Offline Preprocessing completed successfully!   ")
    print("==================================================")

if __name__ == "__main__":
    main()
