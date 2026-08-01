import os
import yaml
import datetime
import time
import cv2
import numpy as np

from logic.heatmap import HeatmapManager
import torch
from logic.risk_model import RiskMLP


class RiskState:
    """
    คลาสเก็บสถานะความเสี่ยงและพฤติกรรมสะสมสำหรับแต่ละ Track ID
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.current_level = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
        self.confidence = 1.0       # ความมั่นใจของการประเมินของ AI (0.0 - 1.0)
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        # รายละเอียดเงื่อนไขพฤติกรรม
        self.is_inside_geofence = False
        self.geofence_name = ""
        self.loitering_detected = False
        self.run_detected = False
        self.weapon_detected = False
        self.path_anomaly_detected = False
        self.trigger_reasons = []
        
        # สำหรับควบคุมการแจ้งเตือนและการบันทึกซ้ำ
        self.last_alert_time = 0.0
        self.alert_logged = False

    def reset_triggers(self):
        self.trigger_reasons = []
        self.is_inside_geofence = False
        self.geofence_name = ""
        self.loitering_detected = False
        self.run_detected = False
        self.weapon_detected = False
        self.path_anomaly_detected = False


    def add_trigger(self, reason: str):
        if reason not in self.trigger_reasons:
            self.trigger_reasons.append(reason)


class RiskAssessmentEngine:
    """
    เอนจิ้นหลักในการประเมินความเสี่ยง ตรวจสอบ Geofence และวิเคราะห์พฤติกรรมสะสม
    """
    def __init__(self, config_path: str = "configs/behavior_config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
        
        # เก็บสถานะความเสี่ยงแยกตามกล้องและ track_id: {camera_name: {track_id: RiskState}}
        self.camera_states = {}
        self.max_age = 5.0  # ลบข้อมูลสะสมหากคนหายไปเกิน 5 วินาที
        
        # เริ่มใช้งานโมดูล Heatmap
        self.heatmap_manager = HeatmapManager(self.config)
        
        # โหลดโมเดลโครงข่ายประสาทเทียมประเมินความเสี่ยง (PyTorch MLP)
        self.risk_model = None
        self.model_meta = None
        self.load_risk_model()

    def load_risk_model(self):
        model_path = "models/risk_classifier.pth"
        meta_path = "models/risk_model_meta.yaml"
        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.model_meta = yaml.safe_load(f)
                
                self.risk_model = RiskMLP(input_dim=6, num_classes=4)
                self.risk_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.risk_model.eval()
                print(f"[RiskAssessmentEngine] Successfully loaded PyTorch Risk Classifier from {model_path}")
            except Exception as e:
                print(f"[RiskAssessmentEngine] Error loading PyTorch model: {e}. Falling back to Rule-based.")
                self.risk_model = None
        else:
            print("[RiskAssessmentEngine] PyTorch weight or metadata files not found. Using Rule-based Fallback.")


    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            print(f"[RiskAssessmentEngine] Loaded config from {self.config_path}")
        else:
            print(f"[RiskAssessmentEngine] Config file not found at {self.config_path}. Using default settings.")
            self.config = {
                "off_hours": {"start": "22:00", "end": "06:00", "debug_always_off_hours": True},
                "thresholds": {"loiter_time_seconds": 8.0, "speed_run_pixels_per_second": 180.0, "weapon_distance_ratio": 0.3},
                "geofences": {},
                "logging": {"csv_path": "data/event_log.csv", "alerts_dir": "data/alerts", "cooldown_seconds": 10.0}
            }

    def is_off_hours(self) -> bool:
        """
        ตรวจสอบว่าปัจจุบันอยู่ในช่วงเวลา Off-Hours หรือไม่
        """
        off_hours_cfg = self.config.get("off_hours", {})
        if off_hours_cfg.get("debug_always_off_hours", False):
            return True  # ✅ ส่งกลับ True เสมอเพื่ออำนวยความสะดวกในการทดสอบ

        start_str = off_hours_cfg.get("start", "22:00")
        end_str = off_hours_cfg.get("end", "06:00")
        
        now = datetime.datetime.now().time()
        try:
            start = datetime.datetime.strptime(start_str, "%H:%M").time()
            end = datetime.datetime.strptime(end_str, "%H:%M").time()
            
            if start <= end:
                return start <= now <= end
            else:
                return now >= start or now <= end
        except Exception as e:
            print(f"[RiskAssessmentEngine] Time parsing error: {e}")
            return False

    def get_geofence_polygons(self, camera_name: str, frame_w: int, frame_h: int) -> list[tuple[str, np.ndarray]]:
        """
        ดึงพิกัด Geofence ของกล้องนั้นๆ และแปลงจากอัตราส่วน (Normalized) เป็นพิกัดพิกเซลจริง
        """
        geofences_cfg = self.config.get("geofences", {})
        camera_geofences = geofences_cfg.get(camera_name, [])
        
        polygons = []
        for gf in camera_geofences:
            name = gf.get("name", "Unknown Zone")
            poly_pts = gf.get("polygon", [])
            
            # แปลงพิกัด 0.0-1.0 เป็นพิกัด pixel จริง
            pts = []
            for pt in poly_pts:
                px = int(pt[0] * frame_w)
                py = int(pt[1] * frame_h)
                pts.append([px, py])
            
            if pts:
                polygons.append((name, np.array(pts, dtype=np.int32)))
        
        return polygons

    def check_geofence_breach(self, camera_name: str, points_to_check: list[tuple[float, float]], frame_w: int, frame_h: int) -> tuple[bool, str]:
        """
        ตรวจสอบลิสต์ของพิกัดพิกเซลต่างๆ (เท้า, เข่า, ลำตัว) ว่ามีจุดใดอยู่ใน Geofence หรือไม่
        """
        polygons = self.get_geofence_polygons(camera_name, frame_w, frame_h)
        for name, poly in polygons:
            # ตรวจสอบจุดทั้งหมดในลิสต์
            for px, py in points_to_check:
                dist = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
                if dist >= 0:
                    return True, name
        return False, ""

    def update_and_assess(
        self,
        camera_name: str,
        result,
        speeds: dict[int, float],
        scale: float,
        frame_w: int,
        frame_h: int,
        current_time: float
    ) -> dict[int, RiskState]:
        """
        ประเมินความเสี่ยงและพฤติกรรมของทุก Detections/Tracks ในแต่ละเฟรมภาพ
        """
        if camera_name not in self.camera_states:
            self.camera_states[camera_name] = {}
        
        active_states = self.camera_states[camera_name]

        # ✅ อัปเดตข้อมูลสะสมความร้อน (Heatmap) ของคาบเวลานี้
        is_off_hours_active = self.is_off_hours()
        self.heatmap_manager.update(
            camera_name, result, scale, frame_w, frame_h, is_off_hours_active, current_time
        )
        
        # เคลียร์ข้อมูลทริกเกอร์ของเฟรมใหม่สำหรับทุก track
        for state in active_states.values():
            state.reset_triggers()


        if result is None or result.boxes is None:
            self._cleanup_stale_tracks(camera_name, current_time)
            return active_states

        boxes = result.boxes
        if boxes.id is None:
            self._cleanup_stale_tracks(camera_name, current_time)
            return active_states

        # แยกประเภท Detections: คน กับ มีด (อาวุธ)
        person_detections = []
        weapon_boxes = []
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            
            # class 0 = person
            if cls_id == 0 and track_id is not None:
                person_detections.append({
                    "track_id": track_id,
                    "xyxy": xyxy,
                    "conf": conf
                })
            # class 43 = knife
            elif cls_id == 43:
                weapon_boxes.append(xyxy)

        # ขีดจำกัดและเงื่อนไขความเสี่ยงจาก Config
        thresholds = self.config.get("thresholds", {})
        loiter_limit = thresholds.get("loiter_time_seconds", 8.0)
        speed_limit = thresholds.get("speed_run_pixels_per_second", 180.0)
        weapon_dist_ratio = thresholds.get("weapon_distance_ratio", 0.3)

        is_off_hours_active = self.is_off_hours()

        # วนรอบคนแต่ละคนเพื่อคำนวณและประเมิน
        for person in person_detections:
            tid = person["track_id"]
            px1, py1, px2, py2 = person["xyxy"]
            
            # ปรับพิกัด YOLO -> พิกัดหน้าจอปัจจุบัน (scale)
            px1_s, py1_s = px1 * scale, py1 * scale
            px2_s, py2_s = px2 * scale, py2 * scale
            
            # พิกัดกึ่งกลางและตำแหน่งเท้าของบุคคล
            p_cx = (px1_s + px2_s) / 2
            p_cy = (py1_s + py2_s) / 2
            p_foot_x = p_cx
            p_foot_y = py2_s  # เท้าอยู่ตำแหน่งล่างสุดของกรอบ
            
            pw = px2_s - px1_s
            ph = py2_s - py1_s
            person_diagonal = np.sqrt(pw**2 + ph**2)

            # ดึงหรือสร้าง RiskState ใหม่
            if tid not in active_states:
                active_states[tid] = RiskState(tid)
            
            state = active_states[tid]
            state.last_seen = current_time

            # ── 1. ตรวจสอบการเดินบุกรุก Geofence (เช็ค เท้า, เข่า, เอว) ───
            p_knee_y = (p_cy + p_foot_y) / 2
            pts_to_check = [
                (p_foot_x, p_foot_y),  # จุดเท้าล่างสุด
                (p_cx, p_knee_y),      # จุดระดับเข่า
                (p_cx, p_cy)           # จุดกึ่งกลางตัว/เอว
            ]
            is_breach, zone_name = self.check_geofence_breach(camera_name, pts_to_check, frame_w, frame_h)
            if is_breach:
                state.is_inside_geofence = True
                state.geofence_name = zone_name
                state.add_trigger(f"In Restricted Area ({zone_name})")

            # ── 2. ตรวจสอบระยะเวลาเดินวนเวียน (Loitering) ─────────────────
            total_duration = current_time - state.first_seen
            if total_duration >= loiter_limit:
                state.loitering_detected = True
                state.add_trigger(f"Loitering ({total_duration:.1f}s)")

            # ── 3. ตรวจสอบความเร็ววิกฤต (Running) ───────────────────────
            speed = speeds.get(tid, 0.0)
            if speed >= speed_limit:
                state.run_detected = True
                state.add_trigger(f"Running ({speed:.0f}px/s)")

            # ── 4. ตรวจจับการถือมีด/อาวุธ (Weapon Association) ───────────
            # ตรวจสอบว่ามีกล่องมีดไหนใกล้พิกัดคนนี้หรือไม่
            for w_xyxy in weapon_boxes:
                wx1, wy1, wx2, wy2 = w_xyxy
                wx1_s, wy1_s = wx1 * scale, wy1 * scale
                wx2_s, wy2_s = wx2 * scale, wy2 * scale
                
                w_cx = (wx1_s + wx2_s) / 2
                w_cy = (wy1_s + wy2_s) / 2
                
                # คำนวณระยะทางจากคนไปมีด
                dist = np.sqrt((p_cx - w_cx)**2 + (p_cy - w_cy)**2)
                
                # หากมีดอยู่ในระยะใกล้พอ (เทียบกับสัดส่วนตัวคน)
                if dist <= (weapon_dist_ratio * person_diagonal):
                    state.weapon_detected = True
                    state.add_trigger("Weapon (Knife) Detected")
                    break  # เจอชิ้นเดียวก็ทริกเกอร์เลย

            # ── 4.5. ตรวจสอบการเดินในเส้นทางแปลกปลอม (Path Anomaly) ───────
            is_anomaly = self.heatmap_manager.check_path_anomaly(
                camera_name, p_cx, p_foot_y, is_off_hours_active, current_time
            )
            if is_anomaly:
                state.path_anomaly_detected = True
                state.add_trigger("Unusual Path Entry")

            # ── 5. ตัดสินใจระดับความเสี่ยง (Sequential State Machine หรือ PyTorch MLP) ──
            if self.risk_model is not None and self.model_meta is not None:
                # ── 5.1. การใช้โครงข่ายประสาทเทียม PyTorch MLP ในการทำนาย ──
                f_geofence = 1.0 if state.is_inside_geofence else 0.0
                f_off_hours = 1.0 if is_off_hours_active else 0.0
                
                # ทำ Normalization/Scaling ฟีเจอร์ต่อเนื่องตาม Metadata ที่เซฟไว้
                norm_cfg = self.model_meta.get("normalization", {})
                loiter_cap = norm_cfg.get("loiter_cap_seconds", 30.0)
                speed_cap = norm_cfg.get("speed_cap_px_per_sec", 500.0)
                
                f_loitering = min(total_duration, loiter_cap) / loiter_cap
                f_speed = min(speed, speed_cap) / speed_cap
                f_weapon = 1.0 if state.weapon_detected else 0.0
                f_anomaly = 1.0 if state.path_anomaly_detected else 0.0
                
                # สร้าง Tensor ขนาด [1, 6]
                features = [f_geofence, f_off_hours, f_loitering, f_speed, f_weapon, f_anomaly]
                features_tensor = torch.tensor([features], dtype=torch.float32)
                
                # รัน Forward pass
                with torch.no_grad():
                    logits = self.risk_model(features_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                    predicted_class = probabilities.argmax().item()
                    confidence = probabilities[predicted_class].item()
                
                # แผนผังคลาส: 0: LOW, 1: MEDIUM, 2: HIGH, 3: CRITICAL
                class_mapping = self.model_meta.get("class_mapping", {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"})
                state.current_level = class_mapping.get(predicted_class, "LOW")
                state.confidence = confidence
            else:
                # ── 5.2. Heuristic Logic Fallback (เมื่อไม่ได้รันเทรนโมเดล) ──
                state.confidence = 1.0
                if state.weapon_detected:
                    state.current_level = "CRITICAL"
                elif state.is_inside_geofence and is_off_hours_active:
                    state.current_level = "CRITICAL"
                    state.add_trigger("Off-Hours Breach")
                elif state.is_inside_geofence and (state.run_detected or state.loitering_detected or state.path_anomaly_detected):
                    state.current_level = "HIGH"
                elif state.is_inside_geofence:
                    state.current_level = "HIGH"
                elif state.run_detected and state.loitering_detected:
                    state.current_level = "HIGH"
                elif state.run_detected or state.loitering_detected or state.path_anomaly_detected:
                    state.current_level = "MEDIUM"
                else:
                    state.current_level = "LOW"


        self._cleanup_stale_tracks(camera_name, current_time)
        return active_states

    def _cleanup_stale_tracks(self, camera_name: str, current_time: float):
        """
        ลบข้อมูลความเสี่ยงของคนเก่าที่หายไปเกินกำหนดเพื่อประหยัดเมมโมรี่
        """
        if camera_name not in self.camera_states:
            return
        
        active_states = self.camera_states[camera_name]
        stale_ids = [
            tid for tid, state in active_states.items()
            if current_time - state.last_seen > self.max_age
        ]
        
        for tid in stale_ids:
            active_states.pop(tid, None)

    # ══════════════════════════════════════════════
    #  DRAW METHODS
    # ══════════════════════════════════════════════
    def draw_geofence(self, frame: np.ndarray, camera_name: str):
        """
        วาดโซน Geofence เป็น Overlay แบบโปร่งแสง
        """
        h, w = frame.shape[:2]
        polygons = self.get_geofence_polygons(camera_name, w, h)
        if not polygons:
            return frame

        overlay = frame.copy()
        is_off_hours_active = self.is_off_hours()
        
        # สีโซน: แดงเข้มถ้าเป็นช่วงเวลาปิดทำการ, น้ำเงินเข้มถ้าเป็นช่วงเวลาปกติ
        if is_off_hours_active:
            zone_color = (0, 0, 150)  # Reddish-brown
            line_color = (0, 0, 255)  # Red
            status_text = "[SECURED - OFF HOURS]"
        else:
            zone_color = (150, 100, 0) # Bluish-grey/Cyan-like
            line_color = (255, 165, 0) # Orange
            status_text = "[ACTIVE - NORMAL HOURS]"

        for name, poly in polygons:
            # วาดโปร่งแสงพื้นที่ใน polygon
            cv2.fillPoly(overlay, [poly], zone_color)
            # วาดขอบพื้นที่
            cv2.polylines(overlay, [poly], True, line_color, 2, cv2.LINE_AA)
            
            # วาดป้ายชื่อโซน
            # หาพิกัดมุมซ้ายบนของโพลีกอนเพื่อปักข้อความ
            rect = cv2.boundingRect(poly)
            label_pos = (rect[0] + 10, rect[1] + 25)
            
            cv2.putText(overlay, f"{name} {status_text}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # รวม Overlay โปร่งแสง 30%
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame

    def draw_risk_overlay(self, frame: np.ndarray, result, scale: float, risk_states: dict[int, RiskState]) -> np.ndarray:
        """
        วาดกรอบ Bounding Box และป้ายสถานะแจ้งเตือนแยกตามระดับความเสี่ยงของคน
        """
        if result is None or result.boxes is None or result.boxes.id is None:
            return frame

        # สีประจำระดับความเสี่ยง (LOW, MEDIUM, HIGH, CRITICAL)
        level_colors = {
            "LOW": (0, 255, 0),        # Green
            "MEDIUM": (0, 255, 255),    # Yellow
            "HIGH": (0, 165, 255),     # Orange
            "CRITICAL": (0, 0, 255)     # Red
        }

        for box, track_id in zip(result.boxes.xyxy, result.boxes.id.int().tolist()):
            cls_id = int(result.boxes.cls[result.boxes.id.int().tolist().index(track_id)].item())
            if cls_id != 0:
                continue # แสดงความเสี่ยงเฉพาะคน

            state = risk_states.get(track_id)
            if state is None:
                continue

            color = level_colors.get(state.current_level, (0, 255, 0))
            
            # ปรับพิกัดแสดงผล
            x1_s = int(box[0] * scale)
            y1_s = int(box[1] * scale)
            x2_s = int(box[2] * scale)
            y2_s = int(box[3] * scale)

            # 1. วาดกล่องตัวคนด้วยสีประเมินความเสี่ยง (หนาขึ้นเมื่อความเสี่ยงเพิ่มขึ้น)
            thickness = 2
            if state.current_level in ["HIGH", "CRITICAL"]:
                thickness = 3
                
                # เพิ่มเอฟเฟกต์มุมกล่อง (Corner brackets) ให้ดูดีมีสไตล์
                length = 20
                # บนซ้าย
                cv2.line(frame, (x1_s, y1_s), (x1_s + length, y1_s), color, 4)
                cv2.line(frame, (x1_s, y1_s), (x1_s, y1_s + length), color, 4)
                # บนขวา
                cv2.line(frame, (x2_s, y1_s), (x2_s - length, y1_s), color, 4)
                cv2.line(frame, (x2_s, y1_s), (x2_s, y1_s + length), color, 4)
                # ล่างซ้าย
                cv2.line(frame, (x1_s, y2_s), (x1_s + length, y2_s), color, 4)
                cv2.line(frame, (x1_s, y2_s), (x1_s, y2_s - length), color, 4)
                # ล่างขวา
                cv2.line(frame, (x2_s, y2_s), (x2_s - length, y2_s), color, 4)
                cv2.line(frame, (x2_s, y2_s), (x2_s, y2_s - length), color, 4)

            cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), color, thickness)

            # 2. วาดป้าย Badge ความเสี่ยงสะสมเหนือศีรษะ
            # แสดงระดับพร้อมสาเหตุทริกเกอร์
            reasons = ", ".join(state.trigger_reasons[:2]) if state.trigger_reasons else "Normal"
            badge_text = f"ID:{track_id} | {state.current_level} ({reasons})"
            
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # วาดพื้นหลังป้าย
            badge_y1 = max(th + 10, y1_s - th - 15)
            cv2.rectangle(frame, (x1_s, badge_y1 - th - 6), (x1_s + tw + 10, badge_y1 + 4), color, -1)
            # ตัวอักษรป้าย (สีดำตัดกับพื้นหลังสีสด)
            text_color = (0, 0, 0)
            cv2.putText(frame, badge_text, (x1_s + 5, badge_y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        return frame
