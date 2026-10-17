import os
import sys
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ai.dataset import ActionVideoDataset, CLASS_NAMES_6CLS, CLASS_NAMES_4CLS, CLASS_NAMES_3CLS
from ai.action_classifier import ActionClassifier

def main():
    parser = argparse.ArgumentParser(description="Advanced Action Classifier Evaluator (Detailed FP, FN, and Explanations)")
    parser.add_argument("--dataset-dir", type=str, 
                        default=r"C:\Users\M S I\.cache\kagglehub\datasets\daudshah\video-dataset\versions\1\dataset-video-split-preprocessed",
                        help="Path to the video dataset root directory")
    parser.add_argument("--schema", type=int, default=3, choices=[3, 4, 6], help="Number of class schema to evaluate (3, 4 or 6)")
    parser.add_argument("--weights", type=str, default=None, 
                        help="Path to trained model weights (default auto-selects based on schema)")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames per video clip")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
    args = parser.parse_args()

    if args.weights is None:
        weights_name = f"models/action_classifier_{args.schema}cls.pth"
    else:
        weights_name = args.weights

    if args.schema == 3:
        txt_test = os.path.join(project_root, "exampleDataset", "test_3cls.txt")
    else:
        txt_test = os.path.join(project_root, "exampleDataset", "test.txt")
    weights_path = os.path.join(project_root, weights_name) if not os.path.isabs(weights_name) else weights_name
    
    if args.schema == 6:
        class_names_dict = CLASS_NAMES_6CLS
    elif args.schema == 4:
        class_names_dict = CLASS_NAMES_4CLS
    else:
        class_names_dict = CLASS_NAMES_3CLS

    print("==========================================================")
    print(f"      Advanced Evaluation Tool ({args.schema} Classes Schema)      ")
    print("==========================================================")
    print(f"Test split file: {txt_test}")
    print(f"Weights file:    {weights_path}")

    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at {weights_path}")
        return

    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:          {device}")

    # 1. Load Test Dataset & DataLoader
    test_dataset = ActionVideoDataset(txt_test, args.dataset_dir, num_frames=args.num_frames, is_training=False, num_classes_schema=args.schema)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # 2. Load Model
    model = ActionClassifier(num_classes=len(class_names_dict), hidden_size=256, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    use_amp = (device.type == "cuda")

    print("\nRunning model inference on test dataset...")
    start_time = time.time()

    with torch.no_grad():
        for step, (frames, labels) in enumerate(test_loader, 1):
            frames = frames.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(frames)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total_eval_time = time.time() - start_time
    print(f"Evaluation completed in {total_eval_time:.1f} seconds.")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 3. Calculate metrics using sklearn
    target_names = [class_names_dict[i] for i in range(len(class_names_dict))]
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "="*80)
    print(" 1. CONFUSION MATRIX (ตารางเมทริกซ์สับสนแสดงการทำนายจริง vs การทำนายของโมเดล)")
    print("="*80)
    title_str = "True \\ Pred"
    header = f"{title_str:<20}" + "".join([f"{name[:12]:>15}" for name in target_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:<20}" + "".join([f"{val:>15}" for val in row])
        print(row_str)
    print("="*80)

    # วาดและบันทึกรูปภาพ Confusion Matrix โดยใช้ scikit-learn ConfusionMatrixDisplay
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=15, values_format='d')
        plt.title(f"Confusion Matrix ({args.schema} Classes Schema)")
        plt.tight_layout()
        
        img_output_path = os.path.join(project_root, "models", f"confusion_matrix_{args.schema}cls.png")
        plt.savefig(img_output_path, dpi=150)
        plt.close()
        print(f"\n[SUCCESS] พล็อตรูปภาพกราฟิกสำเร็จ: บันทึกรูปภาพแล้วที่ -> {img_output_path}")
    except ImportError:
        print("\n[INFO] เพื่อบันทึกภาพ Confusion Matrix เป็นรูปภาพ .png กรุณาติดตั้งไลบรารีเพิ่ม:")
        print("  .\\env\\Scripts\\pip.exe install matplotlib")

    # 4. Advanced Evaluation: FP and FN Calculation per Class
    print("\n" + "="*80)
    print(" 2. DETAILED METRICS BY CLASS (รวมถึงข้อมูลละเอียดของ False Positives & False Negatives)")
    print("="*80)
    
    col_headers = f"{'Class Name':<20}{'Precision':>11}{'Recall':>11}{'F1-Score':>11}{'Support':>9}{'FP (แจ้งพลาด)':>14}{'FN (หลุดตรวจ)':>14}"
    print(col_headers)
    print("-" * len(col_headers))

    # Calculate metrics for each class
    for i in range(len(class_names_dict)):
        name = class_names_dict[i]
        
        # True Positive (TP): Actual is class i, and Predicted is class i
        tp = int(cm[i, i])
        
        # False Positive (FP): Predicted is class i, but Actual is NOT class i
        fp = int(sum(cm[:, i]) - tp)
        
        # False Negative (FN): Actual is class i, but Predicted is NOT class i
        fn = int(sum(cm[i, :]) - tp)
        
        # Support (Actual count)
        support = int(sum(cm[i, :]))
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        row_str = f"{name:<20}{precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>8} {fp:>13} {fn:>13}"
        print(row_str)

    overall_acc = np.mean(all_preds == all_targets) * 100.0
    print("-" * len(col_headers))
    print(f"Overall System Accuracy: {overall_acc:.2f}%")
    print("="*80)

    # 5. Explanations of what each metric means
    print("\n" + "="*80)
    print(" 3. คัมภีร์อธิบายตัวชี้วัดความแม่นยำ (Metric Explanations & Practical Meaning)")
    print("="*80)
    print("""
- [Precision (ความจำเพาะเจาะจง)]
  ความหมาย: โอกาสที่เมื่อโมเดล 'แจ้งเตือน' ว่าเกิดเหตุการณ์นั้นขึ้น แล้วจะเกิดเหตุการณ์นั้นจริง
  การใช้งานจริง: ป้องกันการเกิด False Alarm (ส่งสัญญาณไซเรนเตือนมั่วซั่วจนผู้รักษาความปลอดภัยรำคาญ)
  ตัวอย่าง: Precision 90% หมายถึง โมเดลส่งเสียงไซเรนเตือน 10 ครั้ง จะเป็นเหตุร้ายจริงๆ 9 ครั้ง

- [Recall (ความครอบคลุมในการตรวจจับ)]
  ความหมาย: เปอร์เซ็นต์เหตุร้ายที่เกิดขึ้นจริงทั้งหมด แล้วโมเดลสามารถดักจับตรวจจับได้สำเร็จ
  การใช้งานจริง: บ่งบอกความปลอดภัยของพื้นที่ป้องกัน (ความสามารถในการสกัดและระงับเหตุโดยไม่หลุดรอด)
  ตัวอย่าง: Recall 70% หมายถึง มีเหตุโจรกรรมเกิดขึ้นจริง 10 ครั้ง โมเดลร้องเตือนจับได้ 7 ครั้ง (หลุดรอดไป 3 ครั้ง)

- [F1-Score (ค่าเฉลี่ยความสมดุลประสิทธิผล)]
  ความหมาย: ค่าเฉลี่ยแบบ Harmonic Mean ระหว่าง Precision และ Recall เป็นตัวชี้วัดศักยภาพรวมของคลาสนั้น

- [FP: False Positive (การแจ้งเตือนพลาด)]
  ความหมาย: ภาพหรือวิดีโอที่เป็นเหตุการณ์ปกติ แต่โมเดลดันเข้าใจผิดเตือนภัยว่าเป็นเหตุร้ายแรง
  วิธีแก้ไข: ปรับความมั่นใจ (Threshold) ในการส่งสัญญาณไซเรน หรือเพิ่มวิดีโอ Normal ท่าทางแปลกๆ เข้าชุดฝึก

- [FN: False Negative (การหลุดตรวจจับเหตุร้าย)]
  ความหมาย: มีเหตุร้ายแรงเกิดขึ้นจริง แต่โมเดลมองข้ามและเข้าใจผิดว่าเป็นเหตุการณ์ปกติ
  วิธีแก้ไข: ต้องระวังเป็นพิเศษ! ปรับระบบ Risk Engine ให้ดักจับจาก Geofence / Loitering เพิ่มเติมเพื่อพยุงคะแนน
""")
    print("="*80)

if __name__ == "__main__":
    main()
