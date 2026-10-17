import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def device():
    """Device-agnostic fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_cv2_capture(mocker):
    """Fixture to mock cv2.VideoCapture returning synthetic 720p BGR frames."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Generate synthetic BGR frame (720x1280x3)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    return mock_cap
