import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ai.dataset import ActionVideoDataset, CLASS_NAMES
from ai.action_classifier import ActionClassifier

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Action Classifier with Anti-Overfitting & Label Smoothing")
    parser.add_argument("--dataset-dir", type=str, 
                        default=r"C:\Users\M S I\.cache\kagglehub\datasets\daudshah\video-dataset\versions\1\dataset-video-split",
                        help="Path to the video dataset root directory")
    parser.add_argument("--weights", type=str, default="models/action_classifier.pth", 
                        help="Path to existing trained weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for fine-tuning")
    parser.add_argument("--lr", type=float, default=2e-5, help="Lower learning rate for fine-tuning")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing ratio to prevent overfitting")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience based on validation loss")
    args = parser.parse_args()

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    txt_train = os.path.join(project_root, "exampleDataset", "train.txt")
    txt_valid = os.path.join(project_root, "exampleDataset", "valid.txt")
    weights_path = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights

    print("==================================================")
    print("  Fine-Tuning Action Classifier (Anti-Overfit)   ")
    print("==================================================")
    print(f"Base Weights: {weights_path}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Label Smoothing: {args.label_smoothing} | Monitor Metric: Best Val Loss")

    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Create Datasets & DataLoaders
    print("\nLoading Training & Validation Datasets...")
    train_dataset = ActionVideoDataset(txt_train, args.dataset_dir, num_frames=16, is_training=True)
    valid_dataset = ActionVideoDataset(txt_valid, args.dataset_dir, num_frames=16, is_training=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # 2. Create Model & Load Pre-trained Checkpoint
    print("\nInitializing ActionClassifier...")
    model = ActionClassifier(num_classes=len(CLASS_NAMES), hidden_size=256, pretrained=False)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[SUCCESS] Loaded base model weights from: {weights_path}")
    else:
        print(f"[WARNING] Base weights file not found at {weights_path}. Fine-tuning from random initialization.")

    model = model.to(device)

    # Loss with Label Smoothing & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 3. Fine-Tuning Loop (Monitored by Validation Loss)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    best_model_path = os.path.join(models_dir, "action_classifier.pth")
    meta_path = os.path.join(models_dir, "action_classifier_meta.yaml")

    print("\nStarting Fine-Tuning Loop...")
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, (frames, labels) in enumerate(train_loader, 1):
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * frames.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if step % 50 == 0 or step == len(train_loader):
                batch_acc = (preds == labels).float().mean().item() * 100.0
                print(f"  Fine-tune Epoch [{epoch}/{args.epochs}] | Batch [{step}/{len(train_loader)}] | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.1f}%")

        train_loss = running_loss / max(1, total_train)
        train_acc = (correct_train / max(1, total_train)) * 100.0

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for frames, labels in valid_loader:
                frames = frames.to(device)
                labels = labels.to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(frames)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * frames.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_loss / max(1, total_val)
        val_acc = (correct_val / max(1, total_val)) * 100.0
        
        epoch_time = time.time() - epoch_start
        scheduler.step(val_loss)

        print(f"\n>>> Epoch [{epoch}/{args.epochs}] Summary ({epoch_time:.1f}s):")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")

        # Save Checkpoint if best Val Loss (anti-overfit criteria)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    [CHECKPOINT SAVED] Best Val Loss improved to {best_val_loss:.4f} (Acc: {best_val_acc:.2f}%) -> {best_model_path}\n")
        else:
            patience_counter += 1
            print(f"    (Val Loss did not improve. Best Val Loss remains: {best_val_loss:.4f} | Patience: {patience_counter}/{args.patience})\n")
            if patience_counter >= args.patience:
                print(f"Early Stopping triggered at epoch {epoch} due to no improvement in validation loss.")
                break

    total_time = time.time() - start_train_time
    print("==================================================")
    print(f"Fine-Tuning completed in {total_time/60.0:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.2f}%")
    print("==================================================")

    # Save updated metadata
    meta_info = {
        "model_architecture": "EfficientNet-B0 + GRU (Fine-Tuned)",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "input_frames": 16,
        "image_size": [224, 224],
        "label_smoothing": args.label_smoothing,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "saved_weights": best_model_path
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta_info, f, allow_unicode=True)
    print(f"Saved fine-tuned model metadata to {meta_path}")

if __name__ == "__main__":
    main()
