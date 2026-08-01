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

from ai.dataset import ActionVideoDataset, CLASS_NAMES_3CLS
from ai.action_classifier import ActionClassifier

def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune Action Classifier (3 Classes + Weighted Loss)")
    parser.add_argument("--dataset-dir", type=str, 
                        default=r"C:\Users\M S I\.cache\kagglehub\datasets\daudshah\video-dataset\versions\1\dataset-video-split-preprocessed",
                        help="Path to the preprocessed video dataset root directory")
    parser.add_argument("--weights", type=str, default=None, 
                        help="Optional base weights path to fine-tune from")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames per video clip (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing ratio for anti-overfitting")
    parser.add_argument("--model-name", type=str, default="action_classifier_3cls", help="Base filename for saving model & metadata")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--sampler-alpha", type=float, default=0.5, help="Alpha exponent for smoothing class weights in the sampler")
    args = parser.parse_args()

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    txt_train = os.path.join(project_root, "exampleDataset", "train_3cls.txt")
    txt_valid = os.path.join(project_root, "exampleDataset", "valid_3cls.txt")

    best_model_path = os.path.join(models_dir, f"{args.model_name}.pth")
    meta_path = os.path.join(models_dir, f"{args.model_name}_meta.yaml")

    print("==================================================")
    print("  Training / Fine-tuning Action Classifier (3 Classes)  ")
    print("==================================================")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Frames: {args.num_frames} | LR: {args.lr}")
    print(f"Label Smoothing: {args.label_smoothing} | Anti-Overfit Metric: Best Val Loss & Acc")
    print(f"Saving Checkpoint To: {best_model_path}")

    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Create Datasets with 3 classes
    print("\nLoading Training & Validation Datasets (3 Classes)...")
    train_dataset = ActionVideoDataset(txt_train, args.dataset_dir, num_frames=args.num_frames, is_training=True, num_classes_schema=3)
    valid_dataset = ActionVideoDataset(txt_valid, args.dataset_dir, num_frames=args.num_frames, is_training=False, num_classes_schema=3)

    # 2. Compute Inverse Class Weights for 3 Classes & Setup Sampler
    print("\nCalculating Inverse Class Weights & Setting up Balanced Sampler...")
    class_counts = [0] * len(CLASS_NAMES_3CLS)
    for _, label in train_dataset.video_list:
        class_counts[label] += 1
        
    total_samples = len(train_dataset.video_list)
    class_weights = []
    print("Class distribution in Train Set (3 Classes):")
    for cls_id, count in enumerate(class_counts):
        w = total_samples / (len(CLASS_NAMES_3CLS) * max(1, count))
        class_weights.append(w)
        print(f"  Class {cls_id} ({CLASS_NAMES_3CLS[cls_id]:<16}): {count:>4} samples | Weight: {w:.4f}")

    weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Setup WeightedRandomSampler to solve class imbalance in train batches with alpha smoothing
    sampler_class_weights = []
    for count in class_counts:
        w_smooth = (total_samples / max(1, count)) ** args.sampler_alpha
        sampler_class_weights.append(w_smooth)

    sample_weights = [sampler_class_weights[label] for _, label in train_dataset.video_list]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,  # Shuffles automatically based on weights
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 4. Create Model Architecture & Optionally Load Base Weights
    print("\nInitializing ActionClassifier (3 Classes)...")
    model = ActionClassifier(num_classes=len(CLASS_NAMES_3CLS), hidden_size=256, pretrained=(args.weights is None))
    
    if args.weights:
        weights_file = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights
        if os.path.exists(weights_file):
            # Load state dict but check shape of final classifier fc layer
            state_dict = torch.load(weights_file, map_location=device)
            # If shape is different (e.g. from 4-class or 6-class model), ignore the last layer weight
            if 'fc.weight' in state_dict and state_dict['fc.weight'].shape[0] != len(CLASS_NAMES_3CLS):
                print(f"[INFO] Final layer shape mismatch detected (original: {state_dict['fc.weight'].shape[0]}, target: {len(CLASS_NAMES_3CLS)}). Skipping final fully connected layer weights.")
                del state_dict['fc.weight']
                del state_dict['fc.bias']
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
            print(f"[SUCCESS] Loaded base model weights for fine-tuning from: {weights_file}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        print("Automatic Mixed Precision (FP16) Enabled.")

    # 5. Training Loop
    best_val_loss = float('inf')
    best_val_acc = 0.0

    print("\nStarting Training / Fine-tuning Loop (3 Classes)...")
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
                outputs = model(frames)    # (Batch, 3)
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
                print(f"  Epoch [{epoch}/{args.epochs}] | Batch [{step}/{len(train_loader)}] | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.1f}%")

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

        # Save Checkpoint if best Val Loss or best Val Acc
        if val_loss < best_val_loss or val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
            torch.save(model.state_dict(), best_model_path)
            print(f"    [CHECKPOINT SAVED] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% -> {best_model_path}\n")
        else:
            print(f"    (Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%)\n")

    total_time = time.time() - start_train_time
    print("==================================================")
    print(f"Training completed in {total_time/60.0:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.2f}%")
    print("==================================================")

    # Save metadata
    meta_info = {
        "model_architecture": "EfficientNet-B0 + GRU (3 Classes Super-Consolidated)",
        "num_classes": len(CLASS_NAMES_3CLS),
        "class_names": CLASS_NAMES_3CLS,
        "input_frames": args.num_frames,
        "image_size": [224, 224],
        "class_weights": class_weights,
        "label_smoothing": args.label_smoothing,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "saved_weights": best_model_path
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta_info, f, allow_unicode=True)
    print(f"Saved model metadata to {meta_path}")

if __name__ == "__main__":
    main()
