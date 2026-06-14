# Antigravity Activity Log - AI-Based Smart Surveillance System

| วันเวลา (Timestamp) | หัวข้อที่ทำ (Topic) | รายละเอียดและเหตุผลที่ทำ (Description & Rationale) |
| :--- | :--- | :--- |
| 2026-06-14 19:18:00 | เริ่มต้นระบบบันทึก Log และวิเคราะห์โครงการ | ทำการวิเคราะห์ภาพรวมโปรเจกต์จากซอร์สโค้ดและไฟล์ PDF รายละเอียดข้อกำหนดของ Mini Project เพื่อระบุขั้นตอนการพัฒนาถัดไป และสร้างไฟล์สำหรับบันทึกประวัติการทำงานตามคำสั่งของผู้ใช้ |
| 2026-06-14 19:28:00 | ปรับปรุงแผนงานและออกแบบไฟล์รายละเอียดโครงการ (PDF) | อัปเดตเอกสารแผนงานโครงการเพิ่มฟีเจอร์ระดับสูง (Heatmap, Time-based Geofencing, Sequential Risk State Machine) ตามการแลกเปลี่ยนแนวคิดกับผู้ใช้ และเตรียมพัฒนาสคริปต์สำหรับออกเอกสารสรุปโครงการในรูปแบบ PDF |
| 2026-06-14 19:30:00 | สร้างไฟล์รายละเอียดโครงงาน (PDF) | พัฒนาและเปิดใช้งานสคริปต์ [generate_project_doc.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/generate_project_doc.py) เพื่อดึงและจัดหน้าเอกสารออกมาเป็นไฟล์ PDF [Project_Details_Surveillance.pdf](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/Project_Details_Surveillance.pdf) ภาษาไทยเรียบร้อยแล้ว โดยใช้ฟอนต์ Tahoma ของระบบเพื่อการจัดแสดงผลที่ถูกต้อง |
| 2026-06-14 19:41:00 | พัฒนาระบบวิเคราะห์ความเสี่ยงและแจ้งเตือน (Phase 2) | ออกแบบและพัฒนาโมดูล [risk_assessment.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/logic/risk_assessment.py) และ [alert_manager.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/logic/alert_manager.py) จัดการกำหนดค่าผ่าน [behavior_config.yaml](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/configs/behavior_config.yaml) และรวมเข้ากับ [main.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/main.py) เพื่อตรวจจับ Geofence breach, loitering, speed running และการถืออาวุธ (มีด) พร้อมบันทึกภาพหน้าจอที่มี overlay แจ้งเตือนและประวัติลงไฟล์ CSV |
| 2026-06-14 19:43:00 | ทดสอบรันและจำลองการทำงานระบบ (Integration Run) | ดำเนินการเปิดรัน FastAPI PIR API Server บนพอร์ต 8000 พร้อมทั้งรันโปรแกรมหลัก main.py เพื่อทดสอบระบบประเมินความเสี่ยงและแจ้งเตือน ตรวจสอบการทริกเกอร์เงื่อนไข Off-Hours Geofence Breach และการบันทึกภาพลง alerts ได้สำเร็จและถูกต้อง |
| 2026-06-14 20:44:00 | พัฒนาโปรแกรมวาด Geofence แบบตอบสนอง | พัฒนาสคริปต์เครื่องมือ [geofence_drawer.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/tools/geofence_drawer.py) ให้ผู้ใช้สามารถวาดพื้นที่หวงห้ามด้วยการคลิกเมาส์บนหน้าต่างภาพของกล้องแต่ละตัว และบันทึกพิกัดแบบ Normalized กลับไปยัง [behavior_config.yaml](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/configs/behavior_config.yaml) โดยอัตโนมัติ พร้อมสำรองข้อมูลเดิม ทำการวาดพื้นที่กล้อง 1 สำเร็จ |
| 2026-06-14 20:51:00 | อัปเดตแหล่งสัญญาณกล้อง IP (กล้อง 1) | ทำการสลับและแก้ไขแหล่งสัญญาณของ Camera 1 เป็นกล้อง IP (http://camera_ip/) ในไฟล์ [main.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/main.py) และ [geofence_drawer.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/tools/geofence_drawer.py) พร้อมเขียนคอมเมนต์เก็บบันทึกไฟล์วิดีโอตัวอย่างเดิมสำหรับการสลับเปลี่ยนทดสอบ |
| 2026-06-14 20:52:00 | แก้ไขการเช็คแหล่งสัญญาณในตัววาด Geofence | ปรับเงื่อนไขตรวจสอบใน [geofence_drawer.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/tools/geofence_drawer.py) เพื่อละเว้นการเช็คไฟล์บนดิสก์หากแหล่งสัญญาณกล้องขึ้นต้นด้วย http:// หรือ rtsp:// สำหรับกล้อง IP |
| 2026-06-14 20:54:00 | อัปเดตพารามิเตอร์ URL RTSP ของกล้อง IP | ดำเนินการอัปเดต URL สตรีมกล้อง IP ของผู้ใช้เป็นลิงก์ RTSP (rtsp://admin:password@camera_ip:554) ในไฟล์ [main.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/main.py) และ [geofence_drawer.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/tools/geofence_drawer.py) หลังได้รับความตกลงอนุมัติจากผู้ใช้ |
| 2026-06-14 20:58:00 | ปิดเซิร์ฟเวอร์ PIR API (Uvicorn Server) | ทำการหยุดและปิดการทำงานของ Uvicorn FastAPI Server ที่รันบนพอร์ต 8000 เรียบร้อยแล้วหลังเสร็จสิ้นการรันโปรแกรมหลักและการวาดพื้นที่กับกล้อง IP ของผู้ใช้ |
| 2026-06-14 21:16:00 | ปรับขนาดการแสดงผลของตัววาด Geofence | แก้ไขค่า DISPLAY_WIDTH เป็น 960 ใน [geofence_drawer.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/tools/geofence_drawer.py) เพื่อเพิ่มความละเอียดขนาดหน้าต่างในการเล็งจุดวาดพิกัด Geofence หลังได้รับความตกลงอนุมัติจากผู้ใช้ |
| 2026-06-14 21:31:00 | พัฒนาระบบ Multi-Point Geofencing | แก้ไขฟังก์ชันเช็คโซน in [risk_assessment.py](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/logic/risk_assessment.py) ให้สแกนตรวจสอบ 3 จุดพร้อมกัน (เท้า, เข่า, ลำตัว/เอว) เพื่อแก้ปัญหาคนยืนบังขาช่วงล่างหรือโดนมุมกล้องตัดเท้าขาดหลุดจากโซนเฝ้าระวัง หลังได้รับความตกลงอนุมัติจากผู้ใช้ |
| 2026-06-14 21:37:00 | สร้างคู่มือ Git และอัปเดต Gitignore | จัดทำไฟล์คู่มือทักษะ [git_safe_upload.md](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/skills/git_safe_upload.md) เพื่อกำหนดแนวทางการสลับพาสเวิร์ด/อัปโหลด Git แบบ Conventional Commits และอัปเดต [gitignore](file:///c:/Users/M%20S%20I/Desktop/project_main/ai_surveillance/.gitignore) ให้ละเว้นไฟล์สำรอง .bak |










