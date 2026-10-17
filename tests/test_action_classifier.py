import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from ai.action_classifier import ActionClassifier, ActionClassifierManager
from logic.risk_assessment import RiskAssessmentEngine, RiskState


def test_action_classifier_forward_pass():
    """
    ทดสอบการส่งผ่านข้อมูล (Forward Pass) ของ ActionClassifier โมเดลโครงข่ายประสาท EfficientNet-B0 + GRU
    """
    model = ActionClassifier(num_classes=3, pretrained=False)
    model.eval()

    # Synthetic Input Batch = 2, Seq_len = 16, Channels = 3, Height = 224, Width = 224
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 3)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_action_classifier_manager_init():
    """
    ทดสอบการริเริ่ม ActionClassifierManager และโหลด Metadata
    """
    manager = ActionClassifierManager(
        model_path="models/action_classifier_3cls.pth",
        meta_path="models/action_classifier_3cls_meta.yaml",
        device="cpu"
    )
    assert manager.num_classes == 3
    assert manager.class_names[0] == "Normal"
    assert manager.class_names[1] == "Property_Crime"
    assert manager.class_names[2] == "Personal_Violence"


def test_action_classifier_manager_buffering_and_threshold():
    """
    ทดสอบการจัดเก็บบัฟเฟอร์เฟรม การวิเคราะห์ตามรอบ และการบังคับใช้ Confidence Threshold (70%)
    """
    manager = ActionClassifierManager(
        model_path="models/action_classifier_3cls.pth",
        meta_path="models/action_classifier_3cls_meta.yaml",
        device="cpu",
        conf_threshold=0.70,
        check_interval=1.0
    )

    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # สะสมเฟรมจนครบเงื่อนไข
    for i in range(10):
        cls_id, cls_name, conf = manager.update_and_classify(track_id=1, crop_bgr=crop, current_time=i * 0.1)

    assert 1 in manager.buffers
    assert len(manager.buffers[1]) == 10
    assert isinstance(cls_id, int)
    assert isinstance(cls_name, str)
    assert isinstance(conf, float)


def test_action_classifier_manager_stale_cleanup():
    """
    ทดสอบการลบบัฟเฟอร์ของบุคคลที่หายไปจากฉากกล้อง
    """
    manager = ActionClassifierManager(device="cpu")
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    manager.update_and_classify(track_id=99, crop_bgr=crop, current_time=1.0)
    assert 99 in manager.buffers

    # ลบ track 99 เนื่องจากไม่อยู่ใน active set
    manager.cleanup_stale_tracks(active_track_ids={1, 2, 3})
    assert 99 not in manager.buffers
    assert 99 not in manager.last_results


def test_risk_assessment_action_integration():
    """
    ทดสอบการยกระดับความเสี่ยงเมื่อตรวจพบ Action ชนิด Personal_Violence (Class 2)
    """
    engine = RiskAssessmentEngine()
    
    # สร้าง mock ActionClassifierManager ที่คืนค่า Personal_Violence (conf = 0.85)
    mock_action_manager = MagicMock()
    mock_action_manager.update_and_classify.return_value = (2, "Personal_Violence", 0.85)

    # Mock Detection Result ของ YOLOv8
    mock_box = MagicMock()
    mock_box.__len__.return_value = 1
    mock_box.cls = torch.tensor([0.0])       # Class 0 = Person
    mock_box.conf = torch.tensor([0.90])
    mock_box.xyxy = torch.tensor([[100.0, 100.0, 200.0, 300.0]])
    mock_box.id = torch.tensor([10.0])

    mock_result = MagicMock()
    mock_result.boxes = mock_box

    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    speeds = {10: 10.0}

    states = engine.update_and_assess(
        camera_name="TestCam",
        result=mock_result,
        speeds=speeds,
        scale=1.0,
        frame_w=640,
        frame_h=480,
        current_time=10.0,
        action_manager=mock_action_manager,
        raw_frame=raw_frame
    )

    assert 10 in states
    state = states[10]
    assert state.action_class == 2
    assert state.action_name == "Personal_Violence"
    assert state.action_confidence == 0.85
    assert state.current_level == "CRITICAL"
    assert any("Personal Violence" in reason for reason in state.trigger_reasons)
