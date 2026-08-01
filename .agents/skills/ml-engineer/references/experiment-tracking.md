# Experiment Tracking & Reproducibility Reference

Covers: reproducibility practices, experiment tracking tools, hyperparameter search infrastructure, data/model versioning.

## Table of Contents
1. Reproducibility Checklist
2. Experiment Tracking Tools
3. Data & Model Versioning
4. Hyperparameter Search Infrastructure

---

## 1. Reproducibility Checklist

- [ ] Random seeds set for every source of randomness: Python `random`, NumPy, framework-specific (`torch.manual_seed`, `tf.random.set_seed`), and CUDA (`torch.cuda.manual_seed_all` — note CUDA operations can still be non-deterministic unless `torch.use_deterministic_algorithms(True)` is set, which may cost performance).
- [ ] Library/framework versions pinned and recorded (`requirements.txt`, `pip freeze`, or a lockfile).
- [ ] Data version recorded — a hash, snapshot ID, or immutable path, not just "the current CSV" which can change.
- [ ] Hyperparameters and config logged alongside the resulting model artifact, not just in a script that may change later.
- [ ] Hardware/environment noted if it affects results (e.g. mixed precision behavior differs slightly across GPU generations).

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## 2. Experiment Tracking Tools

| Tool | Good for | Notes |
|---|---|---|
| MLflow | Self-hosted, framework-agnostic tracking + model registry | Popular in enterprise/on-prem setups |
| Weights & Biases (W&B) | Rich dashboards, collaboration, hyperparameter sweeps | Hosted SaaS (or self-hosted); strong visualization |
| TensorBoard | Lightweight, built into PyTorch/TensorFlow | Good for single-run visualization, weaker for comparing many runs |
| Neptune.ai | Similar niche to W&B | Good metadata/versioning features |
| DVC (Data Version Control) | Versioning data/models alongside git | Complements tracking tools, doesn't replace them |

Minimal logging pattern regardless of tool — log per run: config/hyperparameters, metrics per epoch (train + val), the final model artifact or a pointer to it, and the git commit hash of the code that produced it.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"lr": 1e-3, "batch_size": 32})
    for epoch in range(num_epochs):
        # ... training ...
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_artifact("model.pt")
```

## 3. Data & Model Versioning

- Treat datasets as versioned artifacts, not mutable files — a model trained on "data.csv" that later gets silently updated is a reproducibility trap. Tools: DVC, or simply immutable, timestamped/hashed snapshots in object storage.
- Model registry (MLflow Model Registry, W&B Artifacts, or a simple naming/versioning convention) should track: which data version, which code commit, and which hyperparameters produced each model version, plus its evaluation metrics.
- Stage models explicitly (e.g. "staging" vs "production") rather than relying on informal file naming ("model_final_v2_ACTUALLY_final.pt").

## 4. Hyperparameter Search Infrastructure

- **Optuna** — popular, supports pruning of unpromising trials early (important for expensive deep learning runs), integrates with most frameworks.
- **Ray Tune** — good when search needs to be distributed across many machines/GPUs.
- **W&B Sweeps** — convenient if already using W&B for tracking.
- Use **pruning/early stopping within the search** (not just within a single training run) to avoid wasting compute on clearly bad hyperparameter combinations.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    # ... train and evaluate with these hyperparameters ...
    return val_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```
