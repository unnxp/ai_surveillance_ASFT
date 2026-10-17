import os
import cv2
import numpy as np

class HeatmapManager:
    """
    คลาสจัดการและเรนเดอร์แผนภาพสะสมความหนาแน่นความร้อน (Heatmap) แยกช่วงเวลา Day/Night
    เพื่อสนับสนุนการวิเคราะห์เส้นทางเดินอับสายตา (Path Anomaly Detection)
    """
    def __init__(self, config: dict):
        self.config = config.get("heatmap", {})
        self.enabled = self.config.get("enabled", True)
        
        # ดึงการตั้งค่าพารามิเตอร์
        self.short_decay = self.config.get("short_decay", 0.992)
        self.long_decay = self.config.get("long_decay", 0.9999)
        self.radius = self.config.get("radius", 25)
        self.intensity = self.config.get("intensity", 15.0)
        self.alpha_max = self.config.get("alpha", 0.55)
        
        # การตรวจจับเส้นทางเดินผิดปกติ
        self.warmup_seconds = self.config.get("warmup_seconds", 180)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 5.0)
        
        # แมปปิ้ง Colormap ของ OpenCV
        colormap_str = self.config.get("colormap", "COLORMAP_JET").upper()
        self.colormap = getattr(cv2, colormap_str, cv2.COLORMAP_JET)
        
        # โครงสร้างจัดเก็บหน้ากากความร้อนสะสมแยกตามกล้อง: {camera_name: np.ndarray}
        self.masks_short = {}
        self.masks_day = {}
        self.masks_night = {}
        
        # บันทึกเวลาที่เริ่มเห็นคนเป็นครั้งแรกในแต่ละกล้องเพื่อใช้คำนวณ Warmup
        self.first_update_time = {}
        
        # โหมดการแสดงผล (0: OFF, 1: SHORT-TERM, 2: LONG-TERM)
        self.display_mode = 1  # เริ่มต้นที่ SHORT-TERM
        
    def toggle_mode(self) -> str:
        """
        สลับโหมดการแสดงผลแผนที่ความร้อนวนรอบ (Short-Term -> Long-Term -> Off)
        """
        self.display_mode = (self.display_mode + 1) % 3
        modes = {0: "OFF", 1: "SHORT-TERM", 2: "LONG-TERM"}
        return modes[self.display_mode]
        
    def get_display_mode_str(self) -> str:
        modes = {0: "OFF", 1: "SHORT-TERM", 2: "LONG-TERM"}
        return modes[self.display_mode]

    def _ensure_mask_shapes(self, camera_name: str, width: int, height: int):
        """
        ตรวจสอบและสร้าง/ปรับขนาดหน้ากากความร้อนให้ตรงกับขนาดภาพที่ส่งมาประมวลผล
        """
        # สร้างเมื่อไม่มี
        if camera_name not in self.masks_short:
            self.masks_short[camera_name] = np.zeros((height, width), dtype=np.float32)
            self.masks_day[camera_name] = np.zeros((height, width), dtype=np.float32)
            self.masks_night[camera_name] = np.zeros((height, width), dtype=np.float32)
            return

        # ปรับขนาดเมื่อมิติภาพต่างไปจากเดิม
        h_m, w_m = self.masks_short[camera_name].shape
        if h_m != height or w_m != width:
            self.masks_short[camera_name] = cv2.resize(self.masks_short[camera_name], (width, height))
            self.masks_day[camera_name] = cv2.resize(self.masks_day[camera_name], (width, height))
            self.masks_night[camera_name] = cv2.resize(self.masks_night[camera_name], (width, height))

    def update(self, camera_name: str, result, scale: float, width: int, height: int, is_off_hours: bool, current_time: float):
        """
        อัปเดตค่าความร้อนลงบนหน้ากากสะสมแยกโหมดระยะสั้น และระยะยาว (ตามช่วงเวลา Day/Night)
        """
        if not self.enabled:
            return

        self._ensure_mask_shapes(camera_name, width, height)
        
        # บันทึกเวลาอัปเดตแรกสุดของกล้องนี้
        if camera_name not in self.first_update_time:
            self.first_update_time[camera_name] = current_time

        mask_s = self.masks_short[camera_name]
        # เลือกว่าจะสะสมลงหน้ากาก Day หรือ Night ตามสถานะ off_hours
        mask_l = self.masks_night[camera_name] if is_off_hours else self.masks_day[camera_name]

        # 1. ทำการลดทอนความร้อนสะสมตามเวลา (Decay)
        mask_s *= self.short_decay
        
        # หน้ากากระยะยาวจางช้ามาก (ลดลงเฉพาะหน้ากากที่กำลัง Active ในช่วงเวลานั้นๆ)
        mask_l *= self.long_decay
        # ค่อยๆ จืดจางหน้ากากที่ไม่ได้แอคทีฟด้วย แต่อัตราที่ช้าเป็นพิเศษเพื่อไม่ให้ค้างถาวรเกินไป
        inactive_mask = self.masks_day[camera_name] if is_off_hours else self.masks_night[camera_name]
        inactive_mask *= 0.99995

        # 2. ตรวจจับตำแหน่งบุคคลและกระจายความร้อน
        if result is not None and result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id != 0:
                    continue  # ประเมินเฉพาะคลาสบุคคล (0)

                xyxy = boxes.xyxy[i].tolist()
                px1, py1, px2, py2 = xyxy
                
                # แปลงพิกัดสเกลโมเดล -> พิกัดสเกลการแสดงผลปัจจุบัน
                px1_s, py1_s = px1 * scale, py1 * scale
                px2_s, py2_s = px2 * scale, py2 * scale
                
                # หาจุดศูนย์กลางเท้าของบุคคล (กึ่งกลางแกน X, ล่างสุดแกน Y)
                p_cx = int((px1_s + px2_s) / 2)
                p_foot_y = int(py2_s)
                
                # วาดการกระจายความร้อนแบบ Radial Gradient ลงทั้งระยะสั้นและระยะยาว
                self._add_radial_heat(mask_s, p_cx, p_foot_y, self.radius, self.intensity)
                self._add_radial_heat(mask_l, p_cx, p_foot_y, self.radius, self.intensity)

    def _add_radial_heat(self, mask: np.ndarray, cx: int, cy: int, r: int, intensity: float):
        """
        วาดกระจายความร้อนลักษณะ Radial Gradient จากกึ่งกลางไปยังขอบรัศมี (ความร้อนนุ่มนวล)
        """
        h, w = mask.shape
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w - 1, cx + r)
        y2 = min(h - 1, cy + r)

        if x2 <= x1 or y2 <= y1:
            return

        # คำนวณ Grid พิกัดย่อย
        Y, X = np.ogrid[y1 - cy : y2 - cy + 1, x1 - cx : x2 - cx + 1]
        dist_sq = X**2 + Y**2
        r_sq = r**2

        # สูตรกระจายความร้อน: ยิ่งใกล้ยิ่งร้อน (Linear Dropoff)
        heat = (1.0 - np.sqrt(dist_sq) / r) * intensity
        heat[dist_sq > r_sq] = 0.0
        heat[heat < 0.0] = 0.0

        # สะสมค่าความร้อนลงพิกเซล
        mask[y1 : y2 + 1, x1 : x2 + 1] += heat

    def check_path_anomaly(self, camera_name: str, p_cx: float, p_foot_y: float, is_off_hours: bool, current_time: float) -> bool:
        """
        ตรวจสอบว่าตำแหน่งเท้าของบุคคลนั้นอยู่ในพิกัดที่มีความหนาแน่นความร้อนระยะยาวของช่วงเวลานั้นต่ำกว่าเกณฑ์หรือไม่
        (ถ้าต่ำกว่าเกณฑ์ แปลว่าเดินออกนอกเส้นทางสัญจรปกติของช่วงเวลานั้น)
        """
        if not self.enabled or camera_name not in self.masks_long_active(is_off_hours):
            return False

        # 1. เช็คว่าพ้นช่วงเวลาเก็บข้อมูลเริ่มต้น (Warmup) หรือยัง
        start_time = self.first_update_time.get(camera_name, current_time)
        if current_time - start_time < self.warmup_seconds:
            return False  # ยังไม่เปิดใช้งานถ้าอยู่ในช่วงเก็บข้อมูลเริ่มต้นเพื่อป้องกันการแจ้งเตือนผิดพลาด (False Alarm)

        # 2. ดึงหน้ากากความร้อนระยะยาวประจำช่วงเวลา
        mask_l = self.masks_night[camera_name] if is_off_hours else self.masks_day[camera_name]
        h, w = mask_l.shape
        
        # ตรวจสอบขอบพิกัดให้อยู่ในภาพ
        cx = int(np.clip(p_cx, 0, w - 1))
        cy = int(np.clip(p_foot_y, 0, h - 1))

        # ตรวจดูค่าความร้อนรอบๆ จุดนั้น (ขนาดพื้นที่เล็กๆ 5x5 พิกเซล เพื่อความเสถียรลด Noise)
        y1, y2 = max(0, cy - 2), min(h - 1, cy + 2)
        x1, x2 = max(0, cx - 2), min(w - 1, cx + 2)
        
        local_heat = mask_l[y1:y2+1, x1:x2+1].mean()
        
        # คืนค่า True หากความหนาแน่นน้อยกว่าเกณฑ์ที่กำหนด (ถือว่าผิดปกติ)
        return bool(local_heat < self.anomaly_threshold)


    def masks_long_active(self, is_off_hours: bool) -> dict:
        return self.masks_night if is_off_hours else self.masks_day

    def apply_overlay(self, camera_name: str, frame: np.ndarray, is_off_hours: bool) -> np.ndarray:
        """
        เรนเดอร์ภาพสีความร้อนสะสมทับลงในเฟรมภาพวิดีโอ
        """
        if not self.enabled or self.display_mode == 0:
            return frame

        # เลือกหน้ากากที่จะแสดงผลตามโหมดการเปิด (1: Short-Term, 2: Long-Term)
        if self.display_mode == 1:
            if camera_name not in self.masks_short:
                return frame
            mask = self.masks_short[camera_name]
        else:
            mask_l = self.masks_night if is_off_hours else self.masks_day
            if camera_name not in mask_l:
                return frame
            mask = mask_l[camera_name]

        # ── DYNAMIC NORMALIZATION ──────────────────
        # สเกลความหนาแน่นให้อยู่ในช่วง 0-255 สัมพันธ์กับค่าสูงสุด
        # ป้องกันสีล้น (Saturation) เมื่อสะสมความร้อนยาวนาน
        max_val = mask.max()
        norm_scale = max(255.0, max_val)  # คุมให้การแสดงผลไม่ล้น แต่ช่วงเริ่มแรกไม่จ้าเกินไป
        norm_mask = (mask / norm_scale) * 255.0
        norm_mask_uint8 = norm_mask.astype(np.uint8)

        # ── APPLY COLORMAP ────────────────────────
        heatmap_color = cv2.applyColorMap(norm_mask_uint8, self.colormap)

        # ── ALPHA BLENDING ────────────────────────
        # จุดไม่มีคนเดิน (ความร้อน 0) = โปร่งใส 100%
        # จุดมีคนหนาแน่นมาก = ความทึบแสงจำกัดด้วย alpha_max
        alpha_mask = (norm_mask / 255.0) * self.alpha_max
        alpha_mask = np.expand_dims(alpha_mask, axis=2)  # ขยายมิติเป็น (H, W, 1) เพื่อประมวลผล Vector BGR

        # ผสมภาพสูตร: Output = (Original * (1 - Alpha)) + (Heatmap * Alpha)
        blended = frame * (1.0 - alpha_mask) + heatmap_color * alpha_mask
        
        return blended.astype(np.uint8)

    def get_compressed_grid(self, camera_name: str, is_off_hours: bool, grid_size: int = 32) -> list:
        """
        บีบอัดหน้ากากความร้อนสะสมระยะยาวประจำช่วงเวลาให้เหลือเมทริกซ์ Grid ขนาด grid_size x grid_size
        เพื่อประหยัดแบนด์วิดท์ในการส่งสัญญาณไปยังเซิร์ฟเวอร์ส่วนกลาง
        """
        if not self.enabled:
            return [0.0] * (grid_size * grid_size)
            
        mask_l = self.masks_night.get(camera_name) if is_off_hours else self.masks_day.get(camera_name)
        if mask_l is None:
            return [0.0] * (grid_size * grid_size)
            
        # ย่อขนาดของหน้ากากความร้อนด้วยการสุ่มตัวอย่างเฉลี่ย (Area interpolation) เหลือ grid_size x grid_size
        resized = cv2.resize(mask_l, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        
        # ปรับค่าให้อยู่ในสเกลปกติ [0.0, 1.0]
        max_val = resized.max()
        if max_val > 0:
            resized = resized / max_val
            
        # คืนค่าเป็น list ทศนิยมความยาว 1024 (32x32)
        return resized.flatten().tolist()
