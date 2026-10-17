import os

def generate_label_file(src_dir, output_file_path):
    """
    สแกนโฟลเดอร์ SCVD เพื่อตรวจจับไฟล์วิดีโอ (.avi, .mp4) ในโฟลเดอร์ย่อย
    Normal, Violence, Weaponized และส่งออกไฟล์ Text lists พร้อมป้ายกำกับมาตรฐาน
    """
    # นิยามค่า Mapping ป้ายกำกับอิงตาม UCF-Crime มาตรฐาน
    label_mapping = {
        "Normal": 6,        # Normal -> Normal (6)
        "Violence": 12,     # Violence -> Fighting (12)
        "Weaponized": 17    # Weaponized -> Assault (17)
    }

    if not os.path.exists(src_dir):
        print(f"Error: ไม่พบโฟลเดอร์ต้นทางที่: {src_dir}")
        return False

    records = []
    subfolders = ["Normal", "Violence", "Weaponized"]

    for folder in subfolders:
        folder_path = os.path.join(src_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: ไม่พบโฟลเดอร์ย่อย: {folder_path}")
            continue

        label_id = label_mapping[folder]
        files = os.listdir(folder_path)
        
        # ค้นหาเฉพาะไฟล์วิดีโอ
        video_files = [f for f in files if f.lower().endswith(('.avi', '.mp4'))]
        video_files.sort()

        for video_name in video_files:
            # เก็บในรูปแบบ: folder_name/video_name label
            relative_video_path = f"{folder}/{video_name}"
            records.append(f"{relative_video_path} {label_id}")

    # เขียนข้อมูลลงไฟล์ Text
    with open(output_file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r + "\n")

    print(f"[SUCCESS] สร้างไฟล์ฉลากสำเร็จ -> {output_file_path} (รวม {len(records)} รายการ)")
    return True

def main():
    # กำหนดที่อยู่ของโฟลเดอร์หลัก
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scvd_root = os.path.join(project_root, "datasets", "Violence-dataset", "SCVD_converted")
    
    train_src = os.path.join(scvd_root, "Train")
    test_src = os.path.join(scvd_root, "Test")

    output_dir = os.path.join(project_root, "datasets", "Violence-dataset")
    os.makedirs(output_dir, exist_ok=True)

    train_output = os.path.join(output_dir, "SCVD_Trainlist.txt")
    test_output = os.path.join(output_dir, "SCVD_Testlist.txt")

    print("==================================================")
    print("      SCVD Label Annotation File Generator        ")
    print("==================================================")

    generate_label_file(train_src, train_output)
    generate_label_file(test_src, test_output)

    print("==================================================")

if __name__ == "__main__":
    main()
