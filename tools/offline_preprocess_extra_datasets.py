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
    """
    if not bboxes:
        return bboxes
        
    frame_indices = sorted(bboxes.keys())
    first_frame = frame_indices[0]
    last_frame = frame_indices[-1]
    
    for f in range(first_frame + 1, last_frame):
        if f not in bboxes:
            prev_f = max([idx for idx in frame_indices if idx < f])
            next_f = min([idx for idx in frame_indices if idx > f])
            
            weight = (f - prev_f) / (next_f - prev_f)
            box_prev = np.array(bboxes[prev_f])
            box_next = np.array(bboxes[next_f])
            
            box_interp = (1 - weight) * box_prev + weight * box_next
            bboxes[f] = box_interp.tolist()
            
    return bboxes

def letterbox_image(image, target_size=(224, 224)):
    """
    ปรับขนาดรูปภาพโดยรักษาสัดส่วนเดิม และเติมขอบดำกึ่งกลาง (Aspect Ratio Preserved)
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    return canvas

def preprocess_video(video_path, output_path, model, target_size=(224, 224)):
    """
    ตรวจจับคนด้วย YOLOv8 + ByteTrack tracking, ครอปขอบขยาย 50%, ทำ Letterbox และเซฟผลลัพธ์
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
    raw_detections = {}
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frames_cache.append(frame)
        
        # รัน YOLOv8 tracking (ใช้ ByteTrack ทำงานพร้อม Kalman Filter ในตัว)
        results = model.track(frame, persist=True, verbose=False, classes=[0], tracker="bytetrack.yaml")
        
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
        
    # หา ID คนหลักที่ปรากฏนานที่สุด
    primary_id = None
    max_appearances = 0
    for track_id, appearances in raw_detections.items():
        if len(appearances) > max_appearances:
            max_appearances = len(appearances)
            primary_id = track_id
            
    target_bboxes = {}
    if primary_id is not None:
        target_bboxes = raw_detections[primary_id]
        target_bboxes = interpolate_bboxes(target_bboxes)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    last_known_crop = None
    
    for f in range(len(frames_cache)):
        frame = frames_cache[f]
        img_h, img_w, _ = frame.shape
        
        if f in target_bboxes:
            x1, y1, x2, y2 = map(int, target_bboxes[f])
            w = x2 - x1
            h = y2 - y1
            margin_x = int(w * 0.50)
            margin_y = int(h * 0.50)
            
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
            crop = last_known_crop if last_known_crop is not None else frame
            
        crop_letterboxed = letterbox_image(crop, target_size)
        out.write(crop_letterboxed)
        
    out.release()
    return True

def process_youtube_robbery(model, force):
    robbery_root = os.path.join(project_root, "datasets", "Youtube-Robbery-Video-Dataset-master", "Youtube-Robbery-Video-Dataset-master")
    src_dataset_dir = os.path.join(robbery_root, "Dataset")
    dest_dataset_dir = os.path.join(robbery_root, "Dataset-preprocessed")
    
    splits = ["Train-Set", "Test-Set"]
    txt_files = [
        ("Trainlist_Videos_Annotation.txt", "Train-Set"),
        ("Testlist_Videos_Annotation.txt", "Test-Set")
    ]
    
    print("\n>>> Processing Youtube Robbery Video Dataset...")
    for txt_name, split in txt_files:
        txt_path = os.path.join(robbery_root, "Annotation_Files", "Classification", txt_name)
        if not os.path.exists(txt_path):
            print(f"Warning: ไม่พบไฟล์ป้ายกำกับ {txt_path}")
            continue
            
        video_entries = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_entries.append(parts[0])
                    
        print(f"Processing {split} ({len(video_entries)} videos)...")
        
        for video_name in tqdm(video_entries):
            # ตรวจสอบประเภทย่อยเพื่อหาโฟลเดอร์ robbery / norobbery
            subfolder = "norobbery" if "norobbery" in video_name else "robbery"
            
            src_video = os.path.join(src_dataset_dir, split, subfolder, video_name)
            dest_video = os.path.join(dest_dataset_dir, split, subfolder, video_name)
            
            if not os.path.exists(src_video):
                continue
                
            if os.path.exists(dest_video) and not force:
                continue
                
            preprocess_video(src_video, dest_video, model)

def process_scvd(model, force):
    scvd_root = os.path.join(project_root, "datasets", "Violence-dataset")
    src_dataset_dir = os.path.join(scvd_root, "SCVD_converted")
    dest_dataset_dir = os.path.join(scvd_root, "SCVD_converted_preprocessed")
    
    txt_files = [
        ("SCVD_Trainlist.txt", "Train"),
        ("SCVD_Testlist.txt", "Test")
    ]
    
    print("\n>>> Processing SCVD (Violence-dataset)...")
    for txt_name, split in txt_files:
        txt_path = os.path.join(scvd_root, txt_name)
        if not os.path.exists(txt_path):
            print(f"Warning: ไม่พบไฟล์ป้ายกำกับ {txt_path}")
            continue
            
        video_entries = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    video_entries.append(parts[0]) # e.g. "Normal/n001_converted.avi"
                    
        print(f"Processing {split} ({len(video_entries)} videos)...")
        
        for relative_path in tqdm(video_entries):
            src_video = os.path.join(src_dataset_dir, split, relative_path)
            # เปลี่ยนสกุลไฟล์ปลายทางให้เป็น .mp4 เพื่อความเข้ากันได้
            base_name, _ = os.path.splitext(relative_path)
            dest_video = os.path.join(dest_dataset_dir, split, base_name + ".mp4")
            
            if not os.path.exists(src_video):
                continue
                
            if os.path.exists(dest_video) and not force:
                continue
                
            preprocess_video(src_video, dest_video, model)

def main():
    parser = argparse.ArgumentParser(description="Preprocess Youtube Robbery and SCVD datasets (Person Crop Margin 50%)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing preprocessed files")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLOv8 weights for tracking")
    args = parser.parse_args()

    print("==================================================")
    print("  Offline Preprocessing Extra Video Datasets      ")
    print("==================================================")

    print("Loading YOLOv8 tracking model...")
    model = YOLO(args.yolo_weights)

    # 1. รันของ Youtube Robbery Dataset
    process_youtube_robbery(model, args.force)
    
    # 2. รันของ SCVD Dataset
    process_scvd(model, args.force)

    print("\n==================================================")
    print("  All extra datasets preprocessed successfully!   ")
    print("==================================================")

if __name__ == "__main__":
    main()
