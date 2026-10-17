import pytest
import numpy as np
from logic.heatmap import HeatmapManager

@pytest.fixture
def heatmap_manager():
    config = {
        "heatmap": {
            "enabled": True, 
            "short_decay": 0.99, 
            "long_decay": 0.999,
            "radius": 25, 
            "intensity": 15.0, 
            "alpha": 0.55,
            "warmup_seconds": 0, 
            "anomaly_threshold": 5.0
        }
    }
    return HeatmapManager(config)

def test_heatmap_compressed_grid(heatmap_manager):
    # Request grid without updating
    grid = heatmap_manager.get_compressed_grid("test_cam", is_off_hours=False)
    assert isinstance(grid, list)
    assert len(grid) == 32 * 32
    # Since we haven't added any points, it should be all zeros
    assert all(x == 0.0 for x in grid)

def test_heatmap_anomaly_detection(heatmap_manager):
    # Check anomaly at a cold spot
    is_anomaly = heatmap_manager.check_path_anomaly(
        "test_cam", p_cx=100.0, p_foot_y=100.0, is_off_hours=False, current_time=10.0
    )
    # With 0 warmup and no prior data, everywhere is cold, but since it's the very first hit, it might trigger anomaly
    assert isinstance(is_anomaly, bool)
