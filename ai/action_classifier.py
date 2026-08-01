import torch
import torch.nn as nn
import torchvision.models as models

class ActionClassifier(nn.Module):
    """
    โมเดลสถาปัตยกรรมผสม (Hybrid CNN + RNN)
    - CNN: EfficientNet-B0 (สกัดฟีเจอร์ Spatial 2D ทีละเฟรม)
    - RNN: Temporal GRU (เชื่อมความสัมพันธ์ด้านเวลาลากเชื่อม 16 เฟรมเข้าด้วยกัน)
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
        # ขนาดอินพุต x: (Batch_size, Seq_len=16, C=3, H=224, W=224)
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
        # out มิติ: (Batch_size, Seq_len, Hidden_size)
        # h_n มิติ: (Num_layers, Batch_size, Hidden_size)
        out, _ = self.gru(temporal_input)
        
        # เลือกเอาเฉพาะสถานะลับ (Hidden State) ตัวสุดท้ายในลำดับเวลาของแต่ละวิดีโอ (เฟรมที่ 16)
        # ผลลัพธ์มิติ: (Batch_size, Hidden_size)
        last_step_features = out[:, -1, :]
        
        # ส่งเข้า FC classification head
        logits = self.fc(last_step_features) # (Batch_size, Num_classes)
        
        return logits
