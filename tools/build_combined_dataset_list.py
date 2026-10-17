import os
import random

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. ที่อยู่ต้นทางของ UCF-Crime List Files
    ucf_dir = os.path.join(project_root, "exampleDataset")
    train_ucf = os.path.join(ucf_dir, "train.txt")
    valid_ucf = os.path.join(ucf_dir, "valid.txt")
    test_ucf = os.path.join(ucf_dir, "test.txt")

    # 2. ที่อยู่ต้นทางของ Youtube Robbery Dataset
    robbery_root = os.path.join(project_root, "datasets", "Youtube-Robbery-Video-Dataset-master", "Youtube-Robbery-Video-Dataset-master")
    train_robbery = os.path.join(robbery_root, "Annotation_Files", "Classification", "Trainlist_Videos_Annotation.txt")
    test_robbery = os.path.join(robbery_root, "Annotation_Files", "Classification", "Testlist_Videos_Annotation.txt")
    preprocessed_robbery_dir = os.path.join(robbery_root, "Dataset-preprocessed")

    # 3. ที่อยู่ต้นทางของ SCVD (Violence-dataset)
    scvd_root = os.path.join(project_root, "datasets", "Violence-dataset")
    train_scvd = os.path.join(scvd_root, "SCVD_Trainlist.txt")
    test_scvd = os.path.join(scvd_root, "SCVD_Testlist.txt")
    preprocessed_scvd_dir = os.path.join(scvd_root, "SCVD_converted_preprocessed")

    # 4. ที่อยู่ปลายทางสำหรับเซฟไฟล์ลิสต์รวม 3 คลาส
    train_out_path = os.path.join(ucf_dir, "train_3cls.txt")
    valid_out_path = os.path.join(ucf_dir, "valid_3cls.txt")
    test_out_path = os.path.join(ucf_dir, "test_3cls.txt")

    print("==================================================")
    print("      Building Combined 3-Class Dataset List      ")
    print("==================================================")

    # วางระบบตัวแปรเก็บเรคคอร์ดทั้งหมด
    final_train = []
    final_valid = []
    final_test = []

    # --- A. โหลดข้อมูล UCF-Crime เดิม ---
    print("\nLoading UCF-Crime entries...")
    # อ่าน train
    if os.path.exists(train_ucf):
        with open(train_ucf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    final_train.append(line.strip())
    # อ่าน valid
    if os.path.exists(valid_ucf):
        with open(valid_ucf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    final_valid.append(line.strip())
    # อ่าน test
    if os.path.exists(test_ucf):
        with open(test_ucf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    final_test.append(line.strip())

    print(f"  Loaded UCF-Crime: Train={len(final_train)}, Valid={len(final_valid)}, Test={len(final_test)}")

    # --- B. โหลดและแปลงข้อมูล Youtube Robbery ---
    print("\nProcessing Youtube Robbery entries...")
    
    # ดึงข้อมูลจากไฟล์สอน (Trainlist)
    robbery_train_records = []
    if os.path.exists(train_robbery):
        with open(train_robbery, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    video_name = parts[0]
                    label = int(parts[1])
                    # แปลงคลาส: 2 (NoRobbery) -> 6 (Normal UCF-Crime), 1 (Robbery) -> 1 (Stealing UCF-Crime)
                    ucf_label = 6 if label == 2 else 1
                    subfolder = "norobbery" if "norobbery" in video_name else "robbery"
                    
                    # บันทึกเป็นพาธสมบูรณ์ตรงไปยังไฟล์ preprocessed
                    abs_path = os.path.join(preprocessed_robbery_dir, "Train-Set", subfolder, video_name)
                    robbery_train_records.append(f"{abs_path} {ucf_label}")
                    
    # แบ่งข้อมูล 90% เป็น Train และ 10% เป็น Valid
    random.seed(42)
    random.shuffle(robbery_train_records)
    split_idx = int(len(robbery_train_records) * 0.9)
    robbery_train_split = robbery_train_records[:split_idx]
    robbery_valid_split = robbery_train_records[split_idx:]
    
    final_train.extend(robbery_train_split)
    final_valid.extend(robbery_valid_split)

    # ดึงข้อมูลจากไฟล์ทดสอบ (Testlist)
    if os.path.exists(test_robbery):
        with open(test_robbery, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    video_name = parts[0]
                    label = int(parts[1])
                    ucf_label = 6 if label == 2 else 1
                    subfolder = "norobbery" if "norobbery" in video_name else "robbery"
                    
                    abs_path = os.path.join(preprocessed_robbery_dir, "Test-Set", subfolder, video_name)
                    final_test.append(f"{abs_path} {ucf_label}")

    print(f"  Added Youtube Robbery: Train={len(robbery_train_split)}, Valid={len(robbery_valid_split)}, Test={len(robbery_train_records) - split_idx}")

    # --- C. โหลดและแปลงข้อมูล SCVD (Violence-dataset) ---
    print("\nProcessing SCVD (Violence-dataset) entries...")
    
    scvd_train_records = []
    if os.path.exists(train_scvd):
        with open(train_scvd, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rel_path = parts[0]  # e.g. "Normal/n001_converted.avi"
                    ucf_label = int(parts[1]) # 6, 12 หรือ 17
                    
                    # แปลงสกุลไฟล์ปลายทางให้สอดคล้องกับ preprocessed (.mp4)
                    base_name, _ = os.path.splitext(rel_path)
                    abs_path = os.path.join(preprocessed_scvd_dir, "Train", base_name + ".mp4")
                    scvd_train_records.append(f"{abs_path} {ucf_label}")
                    
    # แบ่งข้อมูล 90% เป็น Train และ 10% เป็น Valid
    random.shuffle(scvd_train_records)
    split_idx_scvd = int(len(scvd_train_records) * 0.9)
    scvd_train_split = scvd_train_records[:split_idx_scvd]
    scvd_valid_split = scvd_train_records[split_idx_scvd:]
    
    final_train.extend(scvd_train_split)
    final_valid.extend(scvd_valid_split)

    # ดึงข้อมูลจากไฟล์ทดสอบ (Testlist)
    if os.path.exists(test_scvd):
        with open(test_scvd, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rel_path = parts[0]
                    ucf_label = int(parts[1])
                    
                    base_name, _ = os.path.splitext(rel_path)
                    abs_path = os.path.join(preprocessed_scvd_dir, "Test", base_name + ".mp4")
                    final_test.append(f"{abs_path} {ucf_label}")

    print(f"  Added SCVD: Train={len(scvd_train_split)}, Valid={len(scvd_valid_split)}, Test={len(scvd_train_records) - split_idx_scvd}")

    # --- D. เขียนบันทึกไฟล์ลิสต์ใหม่ลงดิสก์ ---
    print("\nWriting final combined text lists...")
    
    with open(train_out_path, "w", encoding="utf-8") as f:
        for item in final_train:
            f.write(item + "\n")
            
    with open(valid_out_path, "w", encoding="utf-8") as f:
        for item in final_valid:
            f.write(item + "\n")
            
    with open(test_out_path, "w", encoding="utf-8") as f:
        for item in final_test:
            f.write(item + "\n")

    print("\n==================================================")
    print("  Combined lists created successfully!            ")
    print(f"  - Train list: {train_out_path} ({len(final_train)} entries)")
    print(f"  - Valid list: {valid_out_path} ({len(final_valid)} entries)")
    print(f"  - Test list:  {test_out_path} ({len(final_test)} entries)")
    print("==================================================")

if __name__ == "__main__":
    main()
