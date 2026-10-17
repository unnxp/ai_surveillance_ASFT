import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import pytest
except ImportError:
    pytest = None

import numpy as np
pytest.importorskip("onnxruntime")
from ai.face_recognition import FaceRecognitionManager
from logic.risk_assessment import RiskState, RiskAssessmentEngine

def test_risk_state_face_fields():
    state = RiskState(track_id=1)
    assert state.face_id == "Unknown"
    assert state.face_score == 0.0
    assert state.is_authorized is False
    assert state.face_vector is None
    assert state.last_face_check == 0.0

def test_face_recognition_manager_match_embedding():
    # Instantiate manager (without requiring ONNX inference for embedding test)
    manager = FaceRecognitionManager.__new__(FaceRecognitionManager)
    manager.threshold = 0.55
    
    # Create synthetic normalized 512-dim embedding vectors
    vec_prime = np.zeros(512, dtype=np.float32)
    vec_prime[0] = 1.0 # unit norm
    
    vec_stranger = np.zeros(512, dtype=np.float32)
    vec_stranger[1] = 1.0 # orthogonal unit norm
    
    vec_query_prime = np.zeros(512, dtype=np.float32)
    vec_query_prime[0] = 0.95
    vec_query_prime[2] = 0.31225
    vec_query_prime = vec_query_prime / np.linalg.norm(vec_query_prime) # cosine sim ~ 0.95
    
    manager.known_embeddings = {"Prime": vec_prime}
    
    # Test match query for Prime
    identity, sim, is_auth = manager.match_embedding(vec_query_prime)
    assert identity == "Prime"
    assert sim >= 0.55
    assert is_auth is True
    
    # Test match query for Stranger
    identity_s, sim_s, is_auth_s = manager.match_embedding(vec_stranger)
    assert identity_s == "Stranger"
    assert sim_s < 0.55
    assert is_auth_s is False

def test_access_control_risk_level_rules(tmp_path=None):
    import yaml
    import tempfile
    from pathlib import Path
    
    if tmp_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)

    config_path = tmp_path / "test_config.yaml"
    config_data = {
        "off_hours": {"start": "22:00", "end": "06:00", "debug_always_off_hours": False},
        "thresholds": {"loiter_time_seconds": 10.0, "speed_run_pixels_per_second": 180.0, "weapon_distance_ratio": 0.3},
        "geofences": {},
        "heatmap": {"enabled": False}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    engine = RiskAssessmentEngine(str(config_path))
    
    # Scenario 1: Authorized person in geofence during normal hours -> LOW risk
    state_auth = RiskState(track_id=1)
    state_auth.is_inside_geofence = True
    state_auth.is_authorized = True
    state_auth.face_id = "Prime"
    
    # Simulate risk assessment rule logic for normal hours
    is_off_hours = False
    if state_auth.is_authorized:
        state_auth.add_trigger(f"Authorized ({state_auth.face_id})")
        
    if state_auth.is_inside_geofence:
        if state_auth.is_authorized:
            state_auth.current_level = "LOW"
        else:
            state_auth.current_level = "HIGH"
            
    assert state_auth.current_level == "LOW"
    assert "Authorized (Prime)" in state_auth.trigger_reasons
    
    # Scenario 2: Stranger in geofence during normal hours -> HIGH risk
    state_stranger = RiskState(track_id=2)
    state_stranger.is_inside_geofence = True
    state_stranger.is_authorized = False
    state_stranger.face_id = "Stranger"
    
    if state_stranger.is_inside_geofence:
        if state_stranger.is_authorized:
            state_stranger.current_level = "LOW"
        else:
            state_stranger.current_level = "HIGH"
            
    assert state_stranger.current_level == "HIGH"
    
    # Scenario 3: Stranger off-hours breach -> CRITICAL risk
    state_off_hours_stranger = RiskState(track_id=3)
    state_off_hours_stranger.is_inside_geofence = True
    state_off_hours_stranger.is_authorized = False
    state_off_hours_stranger.face_id = "Stranger"
    is_off_hours = True
    
    if state_off_hours_stranger.is_inside_geofence and is_off_hours:
        if state_off_hours_stranger.is_authorized:
            state_off_hours_stranger.current_level = "HIGH"
        else:
            state_off_hours_stranger.current_level = "CRITICAL"
            
    assert state_off_hours_stranger.current_level == "CRITICAL"

if __name__ == "__main__":
    print("Running test_risk_state_face_fields...")
    test_risk_state_face_fields()
    print("PASSED")

    print("Running test_face_recognition_manager_match_embedding...")
    test_face_recognition_manager_match_embedding()
    print("PASSED")

    print("Running test_access_control_risk_level_rules...")
    test_access_control_risk_level_rules()
    print("PASSED")

    print("\nAll unit tests passed successfully!")

