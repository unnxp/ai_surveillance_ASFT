import json
import time
import numpy as np


class EdgePacketExporter:
    """
    โมดูลส่งออกข้อมูล Edge Node เป็น JSON Packet มาตรฐานตามสเปกสถาปัตยกรรม (Phase 4):
    - camera_id, timestamp, is_off_hours, total_tracks
    - tracks: [track_id, bbox, face_vector (512-dim list/null), face_id_local, is_authorized,
               geofence_breach, geofence_name, loiter_duration, weapon_detected, path_anomaly,
               action_class, action_name, action_confidence, trigger_reasons, heuristic_risk_level]
    - heatmap_grid_32x32 (32x32 float array list)
    """

    def __init__(self):
        pass

    def build_packet(
        self,
        camera_name: str,
        risk_states: dict,
        heatmap_grid: np.ndarray | None = None,
        is_off_hours: bool = False,
        current_time: float | None = None
    ) -> dict:
        if current_time is None:
            current_time = time.time()

        tracks_data = []
        for tid, state in risk_states.items():
            # แปลง face_vector (512-dim float list หรือ null)
            face_vec_list = None
            if hasattr(state, "face_vector") and state.face_vector is not None:
                if isinstance(state.face_vector, np.ndarray):
                    face_vec_list = [round(float(v), 6) for v in state.face_vector.tolist()]
                elif isinstance(state.face_vector, (list, tuple)):
                    face_vec_list = [round(float(v), 6) for v in state.face_vector]

            track_info = {
                "track_id": int(tid),
                "bbox": state.bbox if hasattr(state, "bbox") and state.bbox else [],
                "has_face_vector": face_vec_list is not None,
                "face_vector": face_vec_list,
                "geofence_breach": 1 if getattr(state, "is_inside_geofence", False) else 0,
                "geofence_name": getattr(state, "geofence_name", ""),
                "loiter_duration": round(float(current_time - getattr(state, "first_seen", current_time)), 1),
                "weapon_detected": 1 if getattr(state, "weapon_detected", False) else 0,
                "path_anomaly": 1 if getattr(state, "path_anomaly_detected", False) else 0,
                "action_class": getattr(state, "action_class", 0),
                "action_name": getattr(state, "action_name", "Normal"),
                "action_confidence": round(float(getattr(state, "action_confidence", 0.0)), 4),
                "trigger_reasons": getattr(state, "trigger_reasons", [])
            }
            tracks_data.append(track_info)

        # แปลง Heatmap 32x32
        heatmap_list = []
        if heatmap_grid is not None:
            if isinstance(heatmap_grid, np.ndarray):
                heatmap_list = [[round(float(val), 4) for val in row] for row in heatmap_grid.tolist()]
            elif isinstance(heatmap_grid, list):
                heatmap_list = heatmap_grid

        packet = {
            "camera_id": camera_name,
            "timestamp": round(float(current_time), 3),
            "is_off_hours": is_off_hours,
            "total_tracks": len(tracks_data),
            "tracks": tracks_data,
            "heatmap_grid_32x32": heatmap_list
        }
        return packet

    def to_json(
        self,
        camera_name: str,
        risk_states: dict,
        heatmap_grid: np.ndarray | None = None,
        is_off_hours: bool = False,
        current_time: float | None = None,
        indent: int | None = 2
    ) -> str:
        import re
        packet = self.build_packet(camera_name, risk_states, heatmap_grid, is_off_hours, current_time)
        json_str = json.dumps(packet, indent=indent, ensure_ascii=False)
        
        # ยุบเฉพาะอาร์เรย์ตัวเลขบริสุทธิ์ (bbox, face_vector, แถวของ heatmap_grid) ให้อยู่ในบรรทัดเดียว
        def flatten_numeric_array(match):
            content = match.group(0)
            items = re.findall(r'[\d\.\-eE]+|true|false|null', content)
            return '[' + ', '.join(items) + ']'

        json_str = re.sub(r'\[[\s\d\.\-eE,truefalsenull]+\]', flatten_numeric_array, json_str)
        return json_str
