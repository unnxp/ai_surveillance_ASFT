import pytest
import os
import numpy as np
from logic.alert_manager import AlertManager
from logic.risk_assessment import RiskState

@pytest.fixture
def alert_manager(tmp_path):
    csv_path = tmp_path / "event_log.csv"
    alerts_dir = tmp_path / "alerts"
    config = {
        "logging": {
            "csv_path": str(csv_path),
            "alerts_dir": str(alerts_dir),
            "cooldown_seconds": 2.0
        }
    }
    return AlertManager(config)

def test_alert_manager_csv_init(alert_manager):
    assert os.path.exists(alert_manager.csv_path)
    assert os.path.exists(alert_manager.alerts_dir)

def test_check_and_trigger_alert(alert_manager):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a high risk state
    state = RiskState(1)
    state.current_level = "HIGH"
    state.trigger_reasons = ["Loitering"]
    
    risk_states = {1: state}
    
    # Trigger first alert
    alert_manager.check_and_trigger_alert("test_cam", frame, risk_states, current_time=10.0)
    
    assert state.alert_logged is True
    assert state.last_alert_time == 10.0
    
    # Trigger again within cooldown (should be ignored)
    alert_manager.check_and_trigger_alert("test_cam", frame, risk_states, current_time=11.0)
    assert state.last_alert_time == 10.0 # Time didn't update
    
    # Trigger after cooldown (should update)
    alert_manager.check_and_trigger_alert("test_cam", frame, risk_states, current_time=13.0)
    assert state.last_alert_time == 13.0
    
    # Check if files were created
    files = os.listdir(alert_manager.alerts_dir)
    assert len(files) >= 1 # At least one image is saved (might be overwritten if tests run in same second)
    
    with open(alert_manager.csv_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3 # Header + 2 events
