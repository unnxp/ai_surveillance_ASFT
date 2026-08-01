import os
import sys
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ai.dataset import ActionVideoDataset, CLASS_NAMES_6CLS, CLASS_NAMES_4CLS
from ai.action_classifier import ActionClassifier

def main():
    parser = argparse.ArgumentParser(description="Evaluate Action Classifier Performance (6 Classes or 4 Classes)")
    parser.add_argument("--dataset-dir", type=str, 
                        default=r"C:\Users\M S I\.cache\kagglehub\datasets\daudshah\video-dataset\versions\1\dataset-video-split-preprocessed",
                        help="Path to the video dataset root directory")
    parser.add_argument("--schema", type=int, default=4, choices=[4, 6], help="Number of class schema to evaluate (4 or 6)")
    parser.add_argument("--weights", type=str, default=None, 
                        help="Path to trained model weights (default auto-selects based on schema)")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames per video clip")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation")
    args = parser.parse_args()

    if args.weights is None:
        weights_name = f"models/action_classifier_{args.schema}cls.pth"
    else:
        weights_name = args.weights

    txt_test = os.path.join(project_root, "exampleDataset", "test.txt")
    weights_path = os.path.join(project_root, weights_name) if not os.path.isabs(weights_name) else weights_name

    class_names_dict = CLASS_NAMES_6CLS if args.schema == 6 else CLASS_NAMES_4CLS

    print("==================================================")
    print(f"   Evaluating Action Classifier ({args.schema} Classes)   ")
    print("==================================================")
    print(f"Test File: {txt_test}")
    print(f"Model Weights: {weights_path}")
    print(f"Schema: {args.schema} Classes | Frames: {args.num_frames} | Batch Size: {args.batch_size}")

    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at {weights_path}")
        return

    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Test Dataset & DataLoader
    print(f"\nLoading Test Dataset ({args.schema} Classes)...")
    test_dataset = ActionVideoDataset(txt_test, args.dataset_dir, num_frames=args.num_frames, is_training=False, num_classes_schema=args.schema)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # 2. Load Model Architecture & Checkpoint
    print(f"\nLoading ActionClassifier ({args.schema} Classes) Weights...")
    model = ActionClassifier(num_classes=len(class_names_dict), hidden_size=256, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Inference Loop
    all_preds = []
    all_targets = []
    use_amp = (device.type == "cuda")

    print("\nRunning Inference on Test Clips...")
    start_time = time.time()

    with torch.no_grad():
        for step, (frames, labels) in enumerate(test_loader, 1):
            frames = frames.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(frames)
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            if step % 20 == 0 or step == len(test_loader):
                print(f"  Evaluated Batch [{step}/{len(test_loader)}]")

    total_eval_time = time.time() - start_time
    print(f"\nInference completed in {total_eval_time:.1f} seconds.")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 4. Calculate Detailed Metrics
    target_names = [class_names_dict[i] for i in range(len(class_names_dict))]
    
    print("\n" + "="*60)
    print(f"     DETAILED CLASSIFICATION REPORT ({args.schema} CLASSES)    ")
    print("="*60)
    
    report_str = classification_report(
        all_targets, 
        all_preds, 
        target_names=target_names, 
        digits=4,
        zero_division=0
    )
    print(report_str)

    # 5. Confusion Matrix Table
    print("\n" + "="*60)
    print("                    CONFUSION MATRIX                       ")
    print("="*60)
    cm = confusion_matrix(all_targets, all_preds)
    
    title_str = "True \\ Pred"
    header = f"{title_str:<16}" + "".join([f"{name[:12]:>14}" for name in target_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:<16}" + "".join([f"{val:>14}" for val in row])
        print(row_str)
    print("="*60)

if __name__ == "__main__":
    main()
