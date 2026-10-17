import os
import time
import yaml
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ActionClassifier(nn.Module):
    """
    โมเดลสถาปัตยกรรมผสม (Hybrid CNN + RNN)
    - CNN: EfficientNet-B0 (สกัดฟีเจอร์ Spatial 2D ทีละเฟรม)
    - RNN: Temporal GRU (เชื่อมความสัมพันธ์ด้านเวลาลากเชื่อม 16/32 เฟรมเข้าด้วยกัน)
    """
    def __init__(self, num_classes=4, hidden_size=256, num_layers=1, pretrained=True):
        super(ActionClassifier, self).__init__()
        
        # 1. โหลด EfficientNet-B0 Backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
            print("[ActionClassifier] Loaded pre-trained EfficientNet-B0 backbone.")
        else:
            self.backbone = models.efficientnet_b0()
            print("[ActionClassifier] Loaded random-initialized EfficientNet-B0 backbone.")
            
        # ดึงมิติตัวสกัดฟีเจอร์ (สำหรับ EfficientNet-B0 คือ 1280 มิติ)
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # ถอด Classifier Head ชุดเดิมของ ImageNet (1000 คลาส) ออก
        # เพื่อใช้สกัดเอาเฉพาะ Feature Maps ลำดับเฉลี่ย (Adaptive Average Pool)
        self.backbone.classifier = nn.Identity()
        
        # 2. ตั้งค่า Temporal GRU
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 3.Classification Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # ขนาดอินพุต x: (Batch_size, Seq_len=16/32, C=3, H=224, W=224)
        batch_size, seq_len, c, h, w = x.size()
        
        # ยุบมิติ Sequence และ Batch เข้าหากันเพื่อให้ CNN ประมวลผลภาพทีละเฟรมได้พร้อมกัน
        # มิติเปลี่ยนเป็น: (Batch_size * Seq_len, C, H, W)
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        
        # สกัดฟีเจอร์เชิงพื้นที่ด้วย EfficientNet-B0
        spatial_features = self.backbone(x_reshaped) # ผลลัพธ์: (Batch_size * Seq_len, 1280)
        
        # ขยายมิติกลับมาเรียงลำดับเวลาสำหรับ GRU
        # มิติเปลี่ยนเป็น: (Batch_size, Seq_len, 1280)
        temporal_input = spatial_features.view(batch_size, seq_len, -1)
        
        # ประมวลผลลำดับเวลาด้วย GRU
        out, _ = self.gru(temporal_input)
        
        # เลือกเอาเฉพาะสถานะลับ (Hidden State) ตัวสุดท้ายในลำดับเวลาของแต่ละวิดีโอ
        last_step_features = out[:, -1, :]
        
        # ส่งเข้า FC classification head
        logits = self.fc(last_step_features) # (Batch_size, Num_classes)
        
        return logits


class ActionClassifierManager:
    """
    คลาสผู้จัดการสำหรับประมวลผลจำแนกพฤติกรรมบุคคล (Action Classifier) แบบ Real-time:
    - เก็บภาพย่อยของแต่ละ track_id ลงใน Ring Buffer
    - ประมวลผลแบบ Periodic Inference (ทุกๆ ~1.5s) เพื่อประหยัด CPU/GPU VRAM
    - กรองเกณฑ์ความมั่นใจ (Confidence Threshold = 0.70)
    - เคลียร์บัฟเฟอร์ของ track_id ที่ไม่อยู่ในมุมกล้องแล้ว
    """
    def __init__(
        self,
        model_path: str = "models/action_classifier_3cls.pth",
        meta_path: str = "models/action_classifier_3cls_meta.yaml",
        device: str | None = None,
        conf_threshold: float = 0.70,
        check_interval: float = 1.5,
    ):
        self.conf_threshold = conf_threshold
        self.check_interval = check_interval
        
        # 1. กำหนด Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 2. อ่าน Metadata หากมี
        self.num_classes = 3
        self.class_names = {0: "Normal", 1: "Property_Crime", 2: "Personal_Violence"}
        self.input_frames = 32
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                self.num_classes = meta.get("num_classes", 3)
                raw_names = meta.get("class_names", self.class_names)
                # แปลงคีย์ใน class_names ให้เป็น int เสมอ
                self.class_names = {int(k): str(v) for k, v in raw_names.items()}
                self.input_frames = meta.get("input_frames", 32)
                print(f"[ActionClassifierManager] Loaded metadata from {meta_path} ({self.num_classes} classes)")
            except Exception as e:
                print(f"[ActionClassifierManager] Failed to parse {meta_path}: {e}")
                
        # 3. สร้างและโหลดโมเดล
        self.model = ActionClassifier(num_classes=self.num_classes, pretrained=False)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[ActionClassifierManager] Loaded model weights from {model_path}")
            except Exception as e:
                print(f"[ActionClassifierManager] Error loading weights from {model_path}: {e}")
        else:
            print(f"[ActionClassifierManager] Warning: Model weights file not found at {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 4. PyTorch ImageNet Mean / Std on target device (for fast GPU normalization)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # 5. โครงสร้างข้อมูลเก็บบัฟเฟอร์แยกตาม track_id
        # {track_id: deque[np.ndarray (224, 224, 3) uint8]}
        self.buffers: dict[int, deque] = {}
        # {track_id: float} (เวลาที่ทำการวิเคราะห์ล่าสุด)
        self.last_inference_time: dict[int, float] = {}
        # {track_id: (class_id, class_name, confidence)}
        self.last_results: dict[int, tuple[int, str, float]] = {}

    def update_and_classify(
        self,
        track_id: int,
        crop_bgr: np.ndarray,
        current_time: float
    ) -> tuple[int, str, float]:
        """
        สะสมภาพย่อยของบุคคลลงบัฟเฟอร์ และทำการวิเคราะห์พฤติกรรมเมื่อถึงรอบเวลา (Periodic Inference)
        """
        default_res = (0, self.class_names.get(0, "Normal"), 0.0)
        
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
            return self.last_results.get(track_id, default_res)
            
        # 1. แปลง BGR -> RGB และ Resize ด้วย OpenCV (รวดเร็วบน C++ ไม่ดึง CPU)
        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return self.last_results.get(track_id, default_res)
            
        # 2. เพิ่มภาพ uint8 ลงในบัฟเฟอร์ของ track_id นี้
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.input_frames)
            self.last_inference_time[track_id] = 0.0
            self.last_results[track_id] = default_res
            
        buf = self.buffers[track_id]
        buf.append(resized_rgb)
        
        # 3. เช็คเงื่อนไขในการทำ Periodic Inference
        last_check = self.last_inference_time.get(track_id, 0.0)
        time_elapsed = current_time - last_check
        
        # ต้องมีเฟรมสะสมอย่างน้อย 8 เฟรม และพ้นระยะเวลา check_interval
        min_frames = min(8, self.input_frames)
        if len(buf) >= min_frames and (time_elapsed >= self.check_interval or last_check == 0.0):
            # สุ่ม/ขยายเฟรมในบัฟเฟอร์ให้ครบตามจำนวน input_frames (เช่น 32 เฟรม)
            buf_list = list(buf)
            if len(buf_list) == self.input_frames:
                seq_frames = buf_list
            else:
                indices = np.linspace(0, len(buf_list) - 1, self.input_frames, dtype=int)
                seq_frames = [buf_list[i] for i in indices]
                
            # แปลงเป็น Tensor และ Normalize บน GPU แบบ Vectorized (รวดเร็วมาก)
            seq_np = np.array(seq_frames, dtype=np.uint8) # (32, 224, 224, 3)
            tensor_seq = torch.from_numpy(seq_np).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
            tensor_seq = (tensor_seq - self.mean) / self.std
            batch_input = tensor_seq.unsqueeze(0) # (1, 32, 3, 224, 224)
            
            with torch.no_grad():
                logits = self.model(batch_input)
                probs = torch.softmax(logits, dim=1)[0]
                top_class = torch.argmax(probs).item()
                top_conf = probs[top_class].item()
                
            # 4. บังคับใช้ Confidence Threshold (70%)
            if top_conf < self.conf_threshold:
                final_class = 0
                final_name = self.class_names.get(0, "Normal")
            else:
                final_class = top_class
                final_name = self.class_names.get(top_class, "Normal")
                
            result = (final_class, final_name, top_conf)
            self.last_results[track_id] = result
            self.last_inference_time[track_id] = current_time
            
        return self.last_results.get(track_id, default_res)

    def cleanup_stale_tracks(self, active_track_ids: set[int]):
        """
        ลบข้อมูลบัฟเฟอร์ของ track_id ที่ไม่อยู่ในเฟรมแล้วเพื่อประหยัด RAM/GPU Memory
        """
        stored_ids = list(self.buffers.keys())
        for tid in stored_ids:
            if tid not in active_track_ids:
                self.buffers.pop(tid, None)
                self.last_inference_time.pop(tid, None)
                self.last_results.pop(tid, None)

