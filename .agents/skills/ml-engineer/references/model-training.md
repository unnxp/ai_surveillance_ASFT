# Model Development & Training Reference

Covers: architecture selection, training loops, regularization, optimizers/schedulers, distributed/mixed-precision training, common bugs.

## Table of Contents
1. Choosing a Model/Architecture
2. Training Loop Essentials
3. Regularization
4. Optimizers & Learning Rate Schedules
5. Common Training Bugs Checklist
6. Scaling: Mixed Precision & Distributed Training

---

## 1. Choosing a Model/Architecture

Default recommendation order (start simple, add complexity only when justified):

| Data type | Start with | Escalate to if needed |
|---|---|---|
| Tabular | Gradient-boosted trees (XGBoost/LightGBM/CatBoost) or logistic/linear regression as baseline | Neural nets (e.g. TabNet, FT-Transformer) only if trees plateau and data is large |
| Images | Pretrained CNN (ResNet, EfficientNet) or ViT via transfer learning | Train from scratch only with very large in-domain data |
| Text | Pretrained transformer (fine-tune a small/medium LLM or encoder like BERT/RoBERTa) | Custom architectures rarely beat fine-tuned pretrained models |
| Time series | Classical (ARIMA/ETS) or gradient-boosted trees with lag features as baseline | LSTM/Temporal Fusion Transformer/N-BEATS for complex multivariate patterns |
| Small dataset (<1k-10k rows) | Simple models (linear, small trees) — deep learning will overfit | — |

Justify the escalation explicitly: "trees plateaued at X metric, trying Y because..." rather than defaulting to deep learning for its own sake.

## 2. Training Loop Essentials

A correct PyTorch training loop skeleton — use this as a reference for what should be present:

```python
import torch

model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation — model.eval() + no_grad() are easy to forget
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
    val_loss /= len(val_loader.dataset)
    model.train()

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Early stopping / checkpointing should key off val_loss, not train_loss
```

Essentials this skeleton demonstrates:
- `optimizer.zero_grad()` before every backward pass (forgetting this accumulates gradients across batches)
- `model.eval()` + `torch.no_grad()` during validation (forgetting this wastes memory and changes behavior of dropout/batchnorm)
- Switching back to `model.train()` after validation
- Normalizing loss by dataset size for a comparable metric across epochs
- Tracking train *and* val loss every epoch to catch overfitting early

## 3. Regularization

- **Weight decay / L2** — nearly always worth trying, cheap.
- **Dropout** — effective in fully-connected and transformer layers; less standard in conv layers with batch norm.
- **Early stopping** — stop training when val loss stops improving for N epochs (patience). Simple and effective.
- **Data augmentation** — domain-specific (image flips/crops/color jitter; text back-translation/synonym swap; tabular: mixup, noise injection). Often more effective than architectural regularization for small datasets.
- **Batch normalization / layer normalization** — stabilizes training, has a mild regularizing effect. Note: batch norm behaves poorly with very small batch sizes (<8) — prefer layer norm or group norm in that regime.
- **Label smoothing** — helps calibration and generalization in classification, especially with noisy labels.

## 4. Optimizers & Learning Rate Schedules

- **Adam/AdamW** — good default for most deep learning; AdamW (decoupled weight decay) is generally preferred over plain Adam.
- **SGD with momentum** — often generalizes slightly better than Adam for CNNs given enough tuning, at the cost of needing more careful LR tuning.
- **Learning rate schedules** — cosine annealing or linear warmup + decay are strong defaults, especially for transformers (warmup avoids early instability).
- **Learning rate finder** — before committing to a fixed LR, a short LR range test (increase LR exponentially over a few hundred steps and watch loss) locates a good starting range cheaply.
- Batch size and learning rate interact — scaling batch size up generally allows/benefits from scaling LR up roughly proportionally (linear scaling rule), within limits.

## 5. Common Training Bugs Checklist

Run through this when a model "isn't learning" or debugging a training script:

- [ ] Is the loss actually decreasing at all in the first few iterations on a tiny subset of data (can the model overfit 10 examples to ~0 loss)? If not, there's a bug, not a tuning problem.
- [ ] Are labels aligned correctly with inputs (off-by-one, shuffling one but not the other)?
- [ ] Is the loss function appropriate for the task (e.g. `CrossEntropyLoss` expects raw logits, not softmax output, in PyTorch)?
- [ ] Are gradients actually flowing (check for `requires_grad`, frozen layers unintentionally, vanishing/exploding gradients)?
- [ ] Is data normalization/preprocessing applied identically at train and inference time?
- [ ] Is there a train/val leak (see `data-pipeline.md`) making val loss suspiciously good?
- [ ] Is the learning rate too high (loss diverges/NaNs) or too low (loss barely moves)?
- [ ] For classification: are class labels the right dtype (long/int, not float) and correctly zero-indexed for the loss function used?
- [ ] Is randomness seeded consistently enough to distinguish real improvement from noise between runs?

Diagnosing over/underfitting from the train/val gap:
- **High train loss, high val loss** → underfitting: increase model capacity, train longer, reduce regularization, check for bugs suppressing learning.
- **Low train loss, high val loss** → overfitting: add regularization, get more data/augmentation, reduce model capacity, check for train/val leakage inflating the gap further.
- **Val loss lower than train loss** → often means regularization (dropout) is active during training measurement but not validation, or val set is easier/smaller — not usually a sign of a "good" model, worth investigating.

## 6. Scaling: Mixed Precision & Distributed Training

- **Mixed precision (fp16/bf16)** — usually a near-free speedup on modern GPUs with minimal accuracy impact. In PyTorch, use `torch.cuda.amp.autocast()` + `GradScaler` (fp16) or plain `bfloat16` autocast (no scaler needed, more numerically stable, preferred on hardware that supports it, e.g. A100/H100/TPU).
- **Data parallel training** — `DistributedDataParallel` (DDP) in PyTorch is preferred over the older `DataParallel` (better scaling, avoids GIL bottleneck on the main GPU).
- **Gradient accumulation** — simulate a larger effective batch size on limited GPU memory by accumulating gradients over several forward/backward passes before calling `optimizer.step()`.
- Mention these only when the user's described scale (large model, multi-GPU, long training times) actually calls for them — don't add complexity to a small tabular model.
