import os
import csv
import datetime
import threading
import cv2
import numpy as np
from logic.risk_assessment import RiskState

class AlertManager:
    """
    คลาสจัดการการบันทึกเหตุการณ์ (Event Logging) และจับภาพหน้าจอ (Screen Capture)
    เมื่อมีการประเมินระดับความเสี่ยงเป็น HIGH หรือ CRITICAL
    """
    def __init__(self, config: dict):
        self.config = config
        
        # ดึงการตั้งค่า Logging
        log_cfg = self.config.get("logging", {})
        self.csv_path = log_cfg.get("csv_path", "data/event_log.csv")
        self.alerts_dir = log_cfg.get("alerts_dir", "data/alerts")
        self.cooldown = log_cfg.get("cooldown_seconds", 10.0)
        
        self._lock = threading.Lock()
        
        # สร้างโฟลเดอร์สำหรับเก็บภาพแจ้งเตือนหากไม่มี
        os.makedirs(self.alerts_dir, exist_ok=True)
        # สร้างโฟลเดอร์สำหรับเก็บ CSV หากไม่มี
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            
        # สร้างหัวตาราง CSV หากพึ่งเคยรันระบบเป็นครั้งแรก
        self._init_csv()
        print(f"[AlertManager] Initialized. CSV: {self.csv_path} | Alerts Dir: {self.alerts_dir}")

    def _init_csv(self):
        with self._lock:
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp",
                        "Camera",
                        "TrackID",
                        "RiskLevel",
                        "TriggerReason",
                        "ScreenshotPath"
                    ])

    def check_and_trigger_alert(
        self,
        camera_name: str,
        display_frame: np.ndarray,
        risk_states: dict[int, RiskState],
        current_time: float
    ):
        """
        ตรวจสอบสถานะความเสี่ยงของแต่ละ Track
        หากเป็น HIGH หรือ CRITICAL และพ้นระยะ Cooldown หรือระดับความเสี่ยงเพิ่มขึ้น (Escalation)
        จะทำการบันทึก Log ลง CSV และบันทึกรูปภาพ Screenshot ทันที
        """
        for state in risk_states.values():
            # สนใจเฉพาะระดับ HIGH และ CRITICAL
            if state.current_level not in ["HIGH", "CRITICAL"]:
                continue

            # กำหนดคุณลักษณะสำหรับเช็ค Escalation (เช่น จาก HIGH ไป CRITICAL)
            last_logged_level = getattr(state, "last_logged_level", None)
            is_escalation = (
                last_logged_level is not None and 
                self._get_risk_score(state.current_level) > self._get_risk_score(last_logged_level)
            )

            # เช็คว่าควรทริกเกอร์แจ้งเตือนหรือไม่
            elapsed_time = current_time - state.last_alert_time
            should_alert = (
                not state.alert_logged or    # ยังไม่เคยทริกเกอร์เลย
                is_escalation or             # มีการยกระดับระดับความเสี่ยงสูงขึ้น
                elapsed_time >= self.cooldown # พ้นกำหนดเวลาหน่วง Cooldown แล้ว
            )

            if should_alert:
                state.last_alert_time = current_time
                state.alert_logged = True
                state.last_logged_level = state.current_level

                # บันทึกข้อมูลแบบ Thread-safe
                self._process_alert(camera_name, display_frame, state)

    def _get_risk_score(self, level: str) -> int:
        mapping = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return mapping.get(level, 1)

    def _process_alert(self, camera_name: str, frame: np.ndarray, state: RiskState):
        """
        ดำเนินการบันทึก Log ลงไฟล์ CSV และส่งภาพเซฟลงดิสก์
        """
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_time_str = now.strftime("%Y%m%d_%H%M%S")
        
        # ปรับแต่งชื่อไฟล์ให้เป็นมิตรกับระบบปฏิบัติการ (ไม่มีอักขระพิเศษ)
        safe_cam_name = "".join(c if c.isalnum() else "_" for c in camera_name)
        image_name = f"alert_{safe_cam_name}_id{state.track_id}_{file_time_str}.jpg"
        image_path = os.path.join(self.alerts_dir, image_name)

        # บันทึกภาพ Screenshot พร้อม Bounding Box และรายละเอียดความเสี่ยงลงไฟล์รูปภาพ
        # ดึงภาพปัจจุบันมาพ่นข้อความอธิบายเหตุการณ์ในภาพเซฟ
        annotated_frame = frame.copy()
        
        # วาดแถบแจ้งเตือนสีแดงขนาดใหญ่ด้านบนของภาพ Screenshot เพื่อความเด่นชัด
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, 40), (0, 0, 255), -1)
        
        alert_title = f"SECURITY ALERT: {state.current_level} - {camera_name.upper()}"
        cv2.putText(annotated_frame, alert_title, (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # เขียนภาพลงดิสก์
        cv2.imwrite(image_path, annotated_frame)
        
        # บันทึกประวัติเหตุการณ์ใน CSV
        reasons = ", ".join(state.trigger_reasons) if state.trigger_reasons else "N/A"
        
        # แปลง path รูปภาพให้เป็น absolute path หรือ relative เพื่อให้ลิงก์เปิดดูง่าย
        abs_img_path = os.path.abspath(image_path)
        
        with self._lock:
            try:
                with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_str,
                        camera_name,
                        state.track_id,
                        state.current_level,
                        reasons,
                        abs_img_path
                    ])
                print(f"[{state.current_level} ALERT] Saved to log. Track ID: {state.track_id} | Reasons: {reasons}")
            except Exception as e:
                print(f"[AlertManager] Error writing to CSV: {e}")
