import os
import yaml
import datetime
import time
import cv2
import numpy as np

from logic.heatmap import HeatmapManager


class RiskState:
    """
    คลาสเก็บสถานะความเสี่ยงและพฤติกรรมสะสมสำหรับแต่ละ Track ID
    """
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.bbox = []              # [x1, y1, x2, y2]
        self.current_level = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
        self.confidence = 1.0       # ความมั่นใจของการประเมินของ AI (0.0 - 1.0)
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        # ข้อมูลการระบุตัวตนด้วยใบหน้า (Face Recognition / Access Control)
        self.face_id = "Unknown"    # e.g., "Prime", "Stranger", "Unknown"
        self.face_score = 0.0
        self.is_authorized = False
        self.face_vector = None     # 512-dim embedding, nullable
        self.last_face_check = 0.0
        self.face_scan_attempts = 0 # จำนวนครั้งที่สแกนแล้ว (Max 3)
        self.face_locked = False    # ล็อคค่าเมื่อได้ภาพชัด score >= 0.75 หรือครบ 3 ครั้ง

        # ข้อมูลการจำแนกพฤติกรรม (Action Classification)
        self.action_class = 0       # 0=Normal, 1=Property_Crime, 2=Personal_Violence
        self.action_name = "Normal"
        self.action_confidence = 0.0
        self.last_action_check = 0.0

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
        speeds: dict[int, float] | None = None,
        scale: float = 1.0,
        frame_w: int = 640,
        frame_h: int = 480,
        current_time: float = 0.0,
        face_manager=None,
        action_manager=None,
        raw_frame=None,
        check_interval: float = 1.0
    ) -> dict[int, RiskState]:
        if speeds is None:
            speeds = {}
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
        faces_processed_this_frame = 0

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
            state.bbox = [int(px1), int(py1), int(px2), int(py2)]

            # ── 0. Periodic Face Feature Extraction (Smart Multi-Attempt & Batch Spreading) ────────────
            if face_manager is not None and raw_frame is not None:
                if not state.face_locked and state.face_scan_attempts < 3 and faces_processed_this_frame < 1:
                    if (current_time - state.last_face_check >= check_interval):
                        state.last_face_check = current_time
                        h_raw, w_raw = raw_frame.shape[:2]
                        cx1 = max(0, int(px1))
                        cy1 = max(0, int(py1))
                        cx2 = min(w_raw, int(px2))
                        cy2 = min(h_raw, int(py2))
                        
                        if cx2 > cx1 and cy2 > cy1:
                            crop_bgr = raw_frame[cy1:cy2, cx1:cx2]
                            faces_processed_this_frame += 1
                            state.face_scan_attempts += 1
                            
                            if hasattr(face_manager, "extract_face_embedding"):
                                emb, score = face_manager.extract_face_embedding(crop_bgr)
                            else:
                                _, score, _, emb = face_manager.recognize_face_in_crop(crop_bgr)
                            
                            if emb is not None:
                                state.face_vector = emb
                                state.face_score = score
                                state.face_id = "Scanned"
                                # ล็อคค่าหยุดสแกนเมื่อได้ภาพชัด score >= 0.75 หรือสแกนครบ 3 ครั้งแล้ว
                                if score >= 0.75 or state.face_scan_attempts >= 3:
                                    state.face_locked = True
                            else:
                                # สแกนไม่พบใบหน้า (หันหลัง/มุมไกล) เว้นระยะ cooldown 3.0s
                                state.last_face_check = current_time + 2.0

            # ── 0.5. Periodic Action Classification ─────────────────────
            if action_manager is not None and raw_frame is not None:
                h_raw, w_raw = raw_frame.shape[:2]
                cx1 = max(0, int(px1))
                cy1 = max(0, int(py1))
                cx2 = min(w_raw, int(px2))
                cy2 = min(h_raw, int(py2))
                
                if cx2 > cx1 and cy2 > cy1:
                    crop_bgr = raw_frame[cy1:cy2, cx1:cx2]
                    ac_id, ac_name, ac_conf = action_manager.update_and_classify(tid, crop_bgr, current_time)
                    state.action_class = ac_id
                    state.action_name = ac_name
                    state.action_confidence = ac_conf

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

            # ── 4.6. ตรวจสอบพฤติกรรมผิดปกติจาก Action Classifier ─────────
            if state.action_class == 2 and state.action_confidence >= 0.70:
                state.add_trigger(f"Personal Violence ({state.action_confidence*100:.0f}%)")
            elif state.action_class == 1 and state.action_confidence >= 0.70:
                state.add_trigger(f"Property Crime ({state.action_confidence*100:.0f}%)")

            # ── 5. ตัดสินใจระดับความเสี่ยง (Heuristic Rule-Based + Access Control) ─
            state.confidence = 1.0
            if state.is_authorized:
                state.add_trigger(f"Authorized ({state.face_id})")
            elif state.face_id == "Stranger":
                state.add_trigger("Stranger")

            if state.weapon_detected or (state.action_class == 2 and state.action_confidence >= 0.70):
                state.current_level = "CRITICAL"
            elif state.is_inside_geofence and is_off_hours_active:
                if state.is_authorized:
                    state.current_level = "HIGH"
                    state.add_trigger("Off-Hours Entry (Authorized)")
                else:
                    state.current_level = "CRITICAL"
                    state.add_trigger("Off-Hours Intrusion (Stranger)")
            elif (state.action_class == 1 and state.action_confidence >= 0.70) or (state.is_inside_geofence and (state.run_detected or state.loitering_detected or state.path_anomaly_detected)):
                state.current_level = "HIGH"
            elif state.is_inside_geofence:
                if state.is_authorized:
                    state.current_level = "LOW"
                else:
                    state.current_level = "HIGH"
                    state.add_trigger("Restricted Area Stranger")
            elif state.run_detected and state.loitering_detected:
                state.current_level = "HIGH"
            elif state.run_detected or state.loitering_detected or state.path_anomaly_detected:
                state.current_level = "MEDIUM"
            else:
                state.current_level = "LOW"


        self._cleanup_stale_tracks(camera_name, current_time)
        if action_manager is not None:
            active_ids = set(active_states.keys())
            action_manager.cleanup_stale_tracks(active_ids)
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

    def export_edge_packet(self, camera_name: str, current_time: float | None = None, to_json: bool = False, indent: int | None = 2):
        """
        ส่งออกข้อมูล Edge JSON Packet สำหรับส่งต่อให้ Central AI Server (Phase 4 Edge Protocol)
        """
        from logic.edge_protocol import EdgePacketExporter
        exporter = EdgePacketExporter()
        risk_states = self.camera_states.get(camera_name, {})
        heatmap_grid = self.heatmap_manager.get_compressed_grid(camera_name, is_off_hours=self.is_off_hours())
        
        if to_json:
            return exporter.to_json(camera_name, risk_states, heatmap_grid, self.is_off_hours(), current_time, indent=indent)
        return exporter.build_packet(camera_name, risk_states, heatmap_grid, self.is_off_hours(), current_time)

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

            # 2. วาดป้าย Badge ฟีเจอร์ Edge Node เหนือศีรษะ
            face_tag = f" [{state.face_id}]" if state.face_id != "Unknown" else ""
            action_tag = f" [{state.action_name}]" if state.action_name and state.action_name != "Normal" else ""
            geo_tag = f" [{state.geofence_name}]" if state.is_inside_geofence and state.geofence_name else ""
            badge_text = f"ID:{track_id}{face_tag}{action_tag}{geo_tag}"
            
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # วาดพื้นหลังป้าย
            badge_y1 = max(th + 10, y1_s - th - 15)
            cv2.rectangle(frame, (x1_s, badge_y1 - th - 6), (x1_s + tw + 10, badge_y1 + 4), color, -1)
            # ตัวอักษรป้าย (สีดำตัดกับพื้นหลังสีสด)
            text_color = (0, 0, 0)
            cv2.putText(frame, badge_text, (x1_s + 5, badge_y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        return frame
