import os
import sys
import shutil
import cv2
import yaml

# ฟังก์ชันปรับขนาดภาพรักษาอัตราส่วน (เหมือนใน main.py)
def resize_keep_ratio(frame, target_width):
    h, w = frame.shape[:2]
    ratio = target_width / w
    return cv2.resize(frame, (target_width, int(h * ratio)))

def main():
    print("==================================================")
    print("     Interactive Geofence Coordinates Drawer      ")
    print("==================================================")
    
    # 1. เลือกกล้อง
    camera_choice = None
    if len(sys.argv) > 1:
        camera_choice = sys.argv[1].strip()
        
    if not camera_choice:
        print("เลือกกล้องที่ต้องการวาด Geofence:")
        print("1) Camera 1 (Always ON)")
        print("2) Camera 2 (PIR + Self Hold)")
        camera_choice = input("กรอกหมายเลข (1 หรือ 2): ").strip()
    
    if camera_choice in ["1", "cam1", "camera1"]:
        camera_name = "Camera 1 (Always ON)"
        # video_source = "videos/3105196-uhd_3840_2160_30fps.mp4" # ✅ เก็บไว้สำหรับเปลี่ยนกลับมาทดสอบ
        video_source = "rtsp://admin:password@192.168.1.100:554/stream1"
    elif camera_choice in ["2", "cam2", "camera2"]:
        camera_name = "Camera 2 (PIR + Self Hold)"
        video_source = r"C:\Users\M S I\Desktop\project_main\ai_surveillance\videos\test_for_cam2.mp4"
    else:
        print("ตัวเลือกไม่ถูกต้อง ปิดโปรแกรม")
        return

    # 2. เปิดวิดีโอเพื่อดึงเฟรมแรก
    is_url = video_source.startswith("http://") or video_source.startswith("https://") or video_source.startswith("rtsp://")
    if not is_url and not os.path.exists(video_source):
        print(f"Error: ไม่พบไฟล์วิดีโอที่ {video_source}")
        return

    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: ไม่สามารถอ่านเฟรมจากวิดีโอได้")
        return

    # ปรับขนาดเฟรมตามความกว้างใช้งานจริงของสตรีมหลัก (960 พิกเซล เพื่อความละเอียด)
    DISPLAY_WIDTH = 960
    frame = resize_keep_ratio(frame, DISPLAY_WIDTH)
    h, w = frame.shape[:2]
    
    # ตัวแปรเก็บพิกัดจุด (พิกเซลจริงบนหน้าจอแสดงผล)
    points = []

    # 3. เมาส์ Callback สำหรับวาดจุด
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # คลิกซ้าย: เพิ่มจุด
            points.append((x, y))
            print(f"เพิ่มจุด: ({x}, {y}) -> Normalized: [{round(x/w, 3)}, {round(y/h, 3)}]")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # คลิกขวา: ย้อนกลับ (Undo)
            if points:
                popped = points.pop()
                print(f"ลบจุดล่าสุด: {popped}")

    window_name = f"Geofence Drawer - {camera_name}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- วิธีการใช้งานหน้าต่างวาดภาพ ---")
    print("• คลิกซ้าย: วางจุดพื้นที่หวงห้าม")
    print("• คลิกขวา: ลบจุดล่าสุด (Undo)")
    print("• กดปุ่ม 'c': ล้างจุดทั้งหมดและเริ่มต้นใหม่ (Clear)")
    print("• กดปุ่ม 's': บันทึกพิกัดลงไฟล์ behavior_config.yaml (Save)")
    print("• กดปุ่ม 'q' หรือ 'Esc': ปิดโปรแกรมโดยไม่บันทึก (Quit)")
    print("---------------------------------")

    # 4. ลูปแสดงผลและวาดกราฟิกตอบสนอง
    while True:
        temp_frame = frame.copy()
        
        # วาดเส้นเชื่อมและพื้นที่โพลีกอน
        if len(points) > 0:
            # วาดจุดวงกลมแต่ละจุด
            for i, pt in enumerate(points):
                cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(temp_frame, str(i + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            # วาดเส้นเชื่อมจุด
            for i in range(1, len(points)):
                cv2.line(temp_frame, points[i - 1], points[i], (0, 255, 0), 2, cv2.LINE_AA)
            
            # หากมีตั้งแต่ 3 จุดขึ้นไป ให้วาดเส้นปิดพื้นที่เป็นรูปหลายเหลี่ยม
            if len(points) >= 3:
                cv2.line(temp_frame, points[-1], points[0], (0, 165, 255), 1, cv2.LINE_AA)
                # ระบายสีโปร่งแสงอ่อนๆ ในพื้นที่ที่เลือก
                overlay = temp_frame.copy()
                pts_arr = np.array(points, dtype=np.int32)
                cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, temp_frame, 0.8, 0, temp_frame)

        # วาดคู่มือแนะนำที่ด้านล่างภาพ
        cv2.rectangle(temp_frame, (0, h - 25), (w, h), (0, 0, 0), -1)
        cv2.putText(temp_frame, "LeftClick=Add | RightClick=Undo | c=Clear | s=Save | q=Quit",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, temp_frame)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # q หรือ Esc
            print("ปิดโปรแกรมโดยไม่มีการบันทึกข้อมูล")
            break
            
        elif key == ord('c'):  # c (Clear)
            points.clear()
            print("ล้างพิกัดทั้งหมดเริ่มต้นใหม่")
            
        elif key == ord('s'):  # s (Save)
            if len(points) < 3:
                print("Error: ต้องมีอย่างน้อย 3 จุดขึ้นไปในการสร้างโพลีกอนพื้นที่หวงห้าม!")
                continue
                
            # แปลงพิกัดเป็น Normalized Coordinates (0.0 - 1.0)
            normalized_pts = []
            for pt in points:
                nx = round(pt[0] / w, 3)
                ny = round(pt[1] / h, 3)
                normalized_pts.append([nx, ny])
                
            # บันทึกพิกัดลง behavior_config.yaml
            config_path = "configs/behavior_config.yaml"
            if not os.path.exists(config_path):
                print(f"Error: ไม่พบไฟล์ตั้งค่าที่ {config_path}")
                break

            # สำรองไฟล์ Config เดิมก่อน
            backup_path = config_path + ".bak"
            shutil.copy2(config_path, backup_path)
            print(f"สร้างไฟล์สำรองสำเร็จที่: {backup_path}")

            # โหลดข้อมูลเก่า
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # ตรวจสอบและอัปเดตค่าพิกัด
            if "geofences" not in config_data:
                config_data["geofences"] = {}
            if camera_name not in config_data["geofences"]:
                config_data["geofences"][camera_name] = [{"name": "Restricted Area", "polygon": []}]
            
            # เขียนทับที่ดัชนีแรกสุดของโพลีกอนกล้อง
            config_data["geofences"][camera_name][0]["polygon"] = normalized_pts
            
            # บันทึกลงไฟล์ yaml
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
            print(f"\n[SUCCESS] บันทึกพื้นที่ Geofence ใหม่ลง {config_path} เรียบร้อยแล้ว!")
            print("พิกัดใหม่:")
            for pt in normalized_pts:
                print(f"  - {pt}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np
    main()
