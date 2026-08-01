import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ai.face_recognition import ArcFaceEmbedder

def main():
    print("==================================================")
    print("      Face Recognition 1:N Watchlist Evaluation    ")
    print("==================================================")

    # 1. นิยามพาธของไฟล์ข้อมูล
    dataset_dir = os.path.join(project_root, "datasets", "Faces-Dataset")
    csv_path = os.path.join(dataset_dir, "Dataset.csv")
    images_dir = os.path.join(dataset_dir, "Faces", "Faces")

    if not os.path.exists(csv_path):
        print(f"Error: ไม่พบไฟล์ป้ายกำกับที่: {csv_path}")
        return

    # โหลดไฟล์ข้อมูลป้ายกำกับ
    df = pd.read_csv(csv_path)
    print(f"โหลดข้อมูลสำเร็จ: พบรูปภาพใบหน้า {len(df)} รูป จาก Dataset.csv")

    # 2. จัดกลุ่มภาพตามบุคคล (Label Grouping)
    person_images = {}
    for _, row in df.iterrows():
        img_name = row['id']
        label = row['label']
        
        # ตรวจสอบการมีอยู่จริงของไฟล์ภาพ
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        if label not in person_images:
            person_images[label] = []
        person_images[label].append(img_path)

    print(f"จำนวนบุคคลที่มีรูปภาพสมบูรณ์ในระบบ: {len(person_images)} คน")

    # กรองเฉพาะบุคคลที่มีรูปภาพอย่างน้อย 2 รูปขึ้นไป (เพื่อสร้าง 1 Gallery และ >= 1 Query ในการวัดผล)
    valid_people = {name: paths for name, paths in person_images.items() if len(paths) >= 2}
    print(f"จำนวนบุคคลที่นำมาใช้ประเมินผลได้ (มีภาพ >= 2 รูป): {len(valid_people)} คน")

    if len(valid_people) == 0:
        print("Error: ไม่มีบุคคลที่มีรูปภาพเพียงพอสำหรับการทำ 1:N Evaluation (ต้องการ >= 2 รูปต่อคน)")
        return

    # 3. โหลดตัวสกัดฟีเจอร์ใบหน้า (ArcFace Embedder)
    print("\nLoading ArcFace Embedding model (w600k_r50.onnx)...")
    try:
        embedder = ArcFaceEmbedder()
    except Exception as e:
        print(f"Error Loading ArcFace Model: {e}")
        print("กรุณาตรวจสอบว่ามีไฟล์ models/w600k_r50.onnx เรียบร้อยแล้ว")
        return

    # 4. สร้างคลังข้อมูลภาพอ้างอิง (Gallery Database - Watchlist)
    # โดยเลือกภาพที่ 1 ของแต่ละคนมาเป็นตัวอ้างอิง
    gallery_embeddings = []
    gallery_labels = []

    print("\nCreating Gallery Database (Watchlist)...")
    for name, paths in tqdm(valid_people.items(), desc="Extracting Gallery"):
        gallery_img_path = paths[0]
        img = cv2.imread(gallery_img_path)
        if img is None:
            continue
        
        # ปรับขนาดเป็น 112x112 ตามข้อกำหนดของ ArcFace
        img_resized = cv2.resize(img, (112, 112))
        try:
            emb = embedder.get_embedding(img_resized)
            gallery_embeddings.append(emb)
            gallery_labels.append(name)
        except Exception as e:
            continue

    gallery_embeddings = np.array(gallery_embeddings) # Shape: (Num_People, 512)
    print(f"ลงทะเบียนคนใน Watchlist สำเร็จ: {len(gallery_labels)} คน")

    # 5. รันสแกนตรวจสอบความถูกต้อง (Query Verification - 1:N Search)
    # โดยนำภาพที่เหลือทั้งหมดของทุกคนมาเป็น Query มาสืบค้นหาคู่แท้ในคลังข้อมูล
    query_labels_true = []
    query_labels_pred = []
    match_similarities_correct = []
    match_similarities_incorrect = []

    print("\nRunning Watchlist Match Queries...")
    for name, paths in tqdm(valid_people.items(), desc="Matching Queries"):
        query_paths = paths[1:] # ภาพที่ 2 เป็นต้นไป
        
        for q_path in query_paths:
            img = cv2.imread(q_path)
            if img is None:
                continue
                
            img_resized = cv2.resize(img, (112, 112))
            try:
                q_emb = embedder.get_embedding(img_resized) # Shape: (512,)
                
                # คำนวณหาความคล้ายคลึงโคไซน์ (Cosine Similarity) กับทุกใบหน้าใน Watchlist
                # เนื่องจาก ArcFace Embeddings ถูก Normalize เป็น Unit Vector (L2=1) แล้ว
                # การดอทโปรดักต์ (Dot Product) จะเท่ากับ Cosine Similarity ทันที
                similarities = np.dot(gallery_embeddings, q_emb)
                
                # เลือกคนใน Watchlist ที่ได้คะแนนคล้ายคลึงสูงสุด (Top-1 Match)
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                pred_name = gallery_labels[best_match_idx]
                
                query_labels_true.append(name)
                query_labels_pred.append(pred_name)
                
                # เก็บสถิติค่าคะแนน similarity เพื่อวิเคราะห์เกณฑ์ตัดสิน
                if pred_name == name:
                    match_similarities_correct.append(best_similarity)
                else:
                    match_similarities_incorrect.append(best_similarity)
                    
            except Exception as e:
                continue

    # 6. รายงานสถิติประเมินประสิทธิภาพ
    total_queries = len(query_labels_true)
    if total_queries == 0:
        print("Error: ไม่สามารถประมวลผลรูปภาพสืบค้นเพื่อวัดคะแนนได้")
        return

    accuracy = accuracy_score(query_labels_true, query_labels_pred) * 100.0
    avg_sim_correct = np.mean(match_similarities_correct) if match_similarities_correct else 0.0
    avg_sim_incorrect = np.mean(match_similarities_incorrect) if match_similarities_incorrect else 0.0

    print("\n" + "="*80)
    print("                   FACIAL RECOGNITION REPORT SUMMARY")
    print("="*80)
    print(f"Total Watchlist Faces (Gallery Database): {len(gallery_labels)} identities")
    print(f"Total Query Search Attempts:             {total_queries} query images")
    print(f"Top-1 Watchlist Identification Accuracy: {accuracy:.2f}%")
    print(f"Average Similarity of Correct Matches:   {avg_sim_correct:.4f} (Cosine Similarity)")
    print(f"Average Similarity of Incorrect Matches: {avg_sim_incorrect:.4f} (Cosine Similarity)")
    print("="*80)
    
    print("\n[RECOMMENDATION FOR Watchlist Threshold Setting]")
    print(f"ค่าแนะนำสำหรับการตั้งเกณฑ์ตัดสินใจ (Threshold): {(avg_sim_correct + avg_sim_incorrect) / 2:.4f}")
    print("  - หากคะแนนเกินเกณฑ์นี้ระบบจะยอมรับผลลัพธ์การสืบค้นชื่อใบหน้าจริง")
    print("  - หากคะแนนต่ำกว่าเกณฑ์นี้ระบบจะระบุเป็นคนแปลกหน้า (Stranger/Unknown)")
    print("="*80)

if __name__ == "__main__":
    main()
