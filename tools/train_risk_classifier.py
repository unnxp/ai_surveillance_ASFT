import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from logic.risk_model import RiskMLP

# SEED เพื่อให้ผลลัพธ์การสุ่มเหมือนเดิมทุกครั้งที่รันการทดสอบ
np.random.seed(42)
torch.manual_seed(42)

def generate_synthetic_data(num_samples=4000):
    """
    สร้างข้อมูลสังเคราะห์พฤทีชีวะ 6 มิติของบุคคลเพื่อการเรียนรู้
    มิติฟีเจอร์:
      0: is_inside_geofence (0 หรือ 1)
      1: is_off_hours (0 หรือ 1)
      2: loitering_ratio (0.0 ถึง 1.0)
      3: speed_ratio (0.0 ถึง 1.0)
      4: weapon_detected (0 หรือ 1)
      5: path_anomaly (0 หรือ 1)
    ระดับคลาส (ความเสี่ยง):
      - 0: LOW Risk
      - 1: MEDIUM Risk
      - 2: HIGH Risk
      - 3: CRITICAL Risk
    """
    samples_per_class = num_samples // 4
    X = []
    y = []
    
    # ── คลาส 0: LOW ──
    for _ in range(samples_per_class):
        # คนเดินปกติ นอกเขตหวงห้าม ในเวลาปกติ
        is_inside_geofence = 0
        is_off_hours = 0 if np.random.rand() < 0.8 else 1  # กลางคืนก็นอนเฉยๆ ได้
        loitering_ratio = np.random.uniform(0.0, 0.2)
        speed_ratio = np.random.uniform(0.0, 0.3)
        weapon_detected = 0
        path_anomaly = 0 if np.random.rand() < 0.95 else 1
        
        X.append([is_inside_geofence, is_off_hours, loitering_ratio, speed_ratio, weapon_detected, path_anomaly])
        y.append(0)
        
    # ── คลาส 1: MEDIUM ──
    for _ in range(samples_per_class):
        # มีลักษณะเดินลนลาน วิ่งเร็ว หรือเดินวนเวียน นอกเขตหวงห้าม
        is_inside_geofence = 0
        is_off_hours = np.random.choice([0, 1])
        weapon_detected = 0
        
        trigger_type = np.random.choice(["loiter", "speed", "anomaly", "combo"])
        if trigger_type == "loiter":
            loitering_ratio = np.random.uniform(0.3, 0.8)
            speed_ratio = np.random.uniform(0.0, 0.3)
            path_anomaly = 0
        elif trigger_type == "speed":
            loitering_ratio = np.random.uniform(0.0, 0.2)
            speed_ratio = np.random.uniform(0.4, 0.7)
            path_anomaly = 0
        elif trigger_type == "anomaly":
            loitering_ratio = np.random.uniform(0.0, 0.2)
            speed_ratio = np.random.uniform(0.0, 0.3)
            path_anomaly = 1
        else: # combo
            loitering_ratio = np.random.uniform(0.2, 0.5)
            speed_ratio = np.random.uniform(0.3, 0.5)
            path_anomaly = np.random.choice([0, 1])
            
        X.append([is_inside_geofence, is_off_hours, loitering_ratio, speed_ratio, weapon_detected, path_anomaly])
        y.append(1)
        
    # ── คลาส 2: HIGH ──
    for _ in range(samples_per_class):
        # บุกรุกโซนหวงห้ามในช่วงเวลากลางวัน หรือมีพฤติกรรมเสี่ยงมากด้านนอก
        weapon_detected = 0
        
        trigger_type = np.random.choice(["trespass_day", "outer_dangerous_combo"])
        if trigger_type == "trespass_day":
            is_inside_geofence = 1
            is_off_hours = 0
            loitering_ratio = np.random.uniform(0.0, 0.4)
            speed_ratio = np.random.uniform(0.0, 0.4)
            path_anomaly = np.random.choice([0, 1])
        else:
            is_inside_geofence = 0
            is_off_hours = np.random.choice([0, 1])
            # วิ่งพร้อมวนเวียนเป็นระยะเวลานานด้านนอก
            loitering_ratio = np.random.uniform(0.6, 1.0)
            speed_ratio = np.random.uniform(0.6, 1.0)
            path_anomaly = 1
            
        X.append([is_inside_geofence, is_off_hours, loitering_ratio, speed_ratio, weapon_detected, path_anomaly])
        y.append(2)
        
    # ── คลาส 3: CRITICAL ──
    for _ in range(samples_per_class):
        # มีอาวุธ หรือบุกรุกพื้นที่หวงห้ามในเวลากลางคืน (Off-Hours) หรือบุกรุกพร้อมความเร็วสูง
        trigger_type = np.random.choice(["weapon", "trespass_night", "trespass_run"])
        
        if trigger_type == "weapon":
            is_inside_geofence = np.random.choice([0, 1])
            is_off_hours = np.random.choice([0, 1])
            loitering_ratio = np.random.uniform(0.0, 1.0)
            speed_ratio = np.random.uniform(0.0, 1.0)
            weapon_detected = 1
            path_anomaly = np.random.choice([0, 1])
        elif trigger_type == "trespass_night":
            is_inside_geofence = 1
            is_off_hours = 1
            loitering_ratio = np.random.uniform(0.0, 1.0)
            speed_ratio = np.random.uniform(0.0, 1.0)
            weapon_detected = 0
            path_anomaly = np.random.choice([0, 1])
        else: # trespass_run
            is_inside_geofence = 1
            is_off_hours = 0
            loitering_ratio = np.random.uniform(0.0, 1.0)
            speed_ratio = np.random.uniform(0.6, 1.0)  # วิ่งบุกรุกในเขตหวงห้าม
            weapon_detected = 0
            path_anomaly = np.random.choice([0, 1])
            
        X.append([is_inside_geofence, is_off_hours, loitering_ratio, speed_ratio, weapon_detected, path_anomaly])
        y.append(3)
        
    # แปลงเป็น NumPy arrays และสับตำแหน่งข้อมูล (Shuffle) เพื่อให้โมเดลไม่เรียนรู้ตามลำดับการสร้าง
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # เพิ่มสัญญาณรบกวน 1-2% ในข้อมูลต่อเนื่อง และสลับบิตบางตัวเพื่อจำลองเซนเซอร์แกว่ง (เพิ่มความเสถียรของโมเดล)
    for row in range(len(X)):
        # แอดน้อยส์ใน Loitering และ Speed
        X[row, 2] = np.clip(X[row, 2] + np.random.normal(0, 0.02), 0.0, 1.0)
        X[row, 3] = np.clip(X[row, 3] + np.random.normal(0, 0.02), 0.0, 1.0)
        
        # สลับบิต Geofence, Off-hours, Weapon, Anomaly
        for col in [0, 1, 4, 5]:
            if np.random.rand() < 0.01:
                X[row, col] = 1.0 - X[row, col]
                
    return X, y

def main():
    print("==================================================")
    print("      Training Risk Classifier (PyTorch MLP)      ")
    print("==================================================")
    
    # 1. โหลดข้อมูล
    X, y = generate_synthetic_data(num_samples=4000)
    
    # แยกส่วน Train และ Test (80% / 20%)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # แปลงเป็น PyTorch Tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)
    
    # 2. เริ่มต้นสถาปัตยกรรมโมเดลและ Optimizer
    model = RiskMLP(input_dim=6, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 150
    batch_size = 64
    num_batches = len(X_train_t) // batch_size
    
    print(f"Training on {len(X_train_t)} samples, validating on {len(X_test_t)} samples...")
    
    # 3. ลูปการฝึกสอนโมเดล (Training Loop)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # มินิแบตช์ไล่ไป
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # ตรวจสอบค่าความแม่นยำ (Validation) ทุกๆ 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_preds = model(X_train_t).argmax(dim=1)
                train_acc = (train_preds == y_train_t).float().mean().item()
                
                test_preds = model(X_test_t).argmax(dim=1)
                test_acc = (test_preds == y_test_t).float().mean().item()
                
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | Val Acc: {test_acc*100:5.1f}%")
            
    # 4. บันทึกไฟล์โมเดลน้ำหนัก (Weights)
    os.makedirs("models", exist_ok=True)
    model_path = "models/risk_classifier.pth"
    torch.save(model.state_dict(), model_path)
    
    # 5. บันทึก Metadata ของการปรับขนาดข้อมูลและสารบัญคลาส
    meta_path = "models/risk_model_meta.yaml"
    meta_data = {
        "features": [
            "is_inside_geofence",
            "is_off_hours",
            "loitering_ratio",
            "speed_ratio",
            "weapon_detected",
            "path_anomaly_detected"
        ],
        "class_mapping": {
            0: "LOW",
            1: "MEDIUM",
            2: "HIGH",
            3: "CRITICAL"
        },
        "normalization": {
            "loitering_cap_seconds": 30.0,
            "speed_cap_px_per_sec": 500.0
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    print("==================================================")
    print(f"[SUCCESS] Trained model saved to: {model_path}")
    print(f"[SUCCESS] Model metadata saved to: {meta_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
