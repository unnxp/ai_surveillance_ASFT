import json
import pytest
import numpy as np
from unittest.mock import MagicMock

from logic.edge_protocol import EdgePacketExporter
from logic.risk_assessment import RiskAssessmentEngine, RiskState


def test_edge_packet_exporter_structure():
    """
    ทดสอบโครงสร้างข้อมูล Edge JSON Packet (camera_id, timestamp, tracks, heatmap_grid_32x32)
    """
    exporter = EdgePacketExporter()
    
    # สร้าง RiskState จำลองที่มี 512-dim face_vector
    state = RiskState(track_id=1)
    state.bbox = [100, 150, 250, 350]
    state.face_id = "Prime"
    state.is_authorized = True
    state.face_vector = np.random.randn(512).astype(np.float32)
    state.is_inside_geofence = True
    state.geofence_name = "Restricted Zone A"
    state.loitering_detected = True
    state.action_class = 1
    state.action_name = "Property_Crime"
    state.action_confidence = 0.82
    state.trigger_reasons = ["In Restricted Area", "Property Crime (82%)"]
    state.current_level = "HIGH"

    risk_states = {1: state}
    heatmap_32x32 = np.ones((32, 32), dtype=np.float32) * 0.5

    packet = exporter.build_packet(
        camera_name="TestCam1",
        risk_states=risk_states,
        heatmap_grid=heatmap_32x32,
        is_off_hours=True,
        current_time=1000.0
    )

    assert packet["camera_id"] == "TestCam1"
    assert packet["timestamp"] == 1000.0
    assert packet["is_off_hours"] is True
    assert packet["total_tracks"] == 1

    track = packet["tracks"][0]
    assert track["track_id"] == 1
    assert track["bbox"] == [100, 150, 250, 350]
    assert track["has_face_vector"] is True
    assert isinstance(track["face_vector"], list)
    assert len(track["face_vector"]) == 512
    assert track["geofence_breach"] == 1
    assert track["geofence_name"] == "Restricted Zone A"
    assert track["action_class"] == 1
    assert track["action_name"] == "Property_Crime"
    assert track["action_confidence"] == 0.82

    # ตรวจสอบ Heatmap 32x32
    assert len(packet["heatmap_grid_32x32"]) == 32
    assert len(packet["heatmap_grid_32x32"][0]) == 32


def test_edge_packet_exporter_null_face_vector():
    """
    ทดสอบการส่งออกเมื่อยังไม่สามารถสแกนใบหน้าได้ (face_vector = null)
    """
    exporter = EdgePacketExporter()
    state = RiskState(track_id=2)
    state.face_vector = None
    
    packet = exporter.build_packet("CamNoFace", {2: state}, current_time=500.0)
    track = packet["tracks"][0]
    assert track["face_vector"] is None


def test_edge_packet_json_string_validity():
    """
    ทดสอบการแปลงเป็นสตริง JSON ที่ถูกต้องตามมาตรฐาน
    """
    exporter = EdgePacketExporter()
    state = RiskState(track_id=3)
    json_str = exporter.to_json("CamJSON", {3: state}, current_time=100.0)
    
    # สามารถ parse กลับเป็น dict ได้อย่างถูกต้อง
    parsed = json.loads(json_str)
    assert parsed["camera_id"] == "CamJSON"
    assert parsed["total_tracks"] == 1


def test_risk_assessment_engine_export_integration():
    """
    ทดสอบการเรียกใช้งาน export_edge_packet ผ่าน RiskAssessmentEngine
    """
    engine = RiskAssessmentEngine()
    
    # เพิ่ม RiskState
    state = RiskState(track_id=5)
    engine.camera_states["TestCam"] = {5: state}
    
    json_output = engine.export_edge_packet("TestCam", current_time=200.0, to_json=True)
    assert isinstance(json_output, str)
    
    parsed = json.loads(json_output)
    assert parsed["camera_id"] == "TestCam"
    assert parsed["total_tracks"] == 1
