# Frameworks Cheatsheet & Deployment Checklist

Quick syntax lookups across common frameworks, plus a lightweight deployment/monitoring checklist. This skill covers deployment at a practical/starting-checklist level, not deep MLOps/infra.

## Table of Contents
1. scikit-learn
2. XGBoost / LightGBM
3. PyTorch
4. TensorFlow / Keras
5. Deployment Checklist (light)
6. Monitoring Checklist (light)

---

## 1. scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Pipeline pattern avoids leakage automatically — fit/transform happens
# correctly inside cross_val_score or GridSearchCV
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
```
Prefer `Pipeline`/`ColumnTransformer` over manual fit/transform calls — it structurally prevents the "fit on full data" leakage bug.

## 2. XGBoost / LightGBM

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=imbalance_ratio,  # for imbalanced classification
    early_stopping_rounds=30,
    eval_metric="auc",
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
```
Key tunables in rough order of impact: `learning_rate` + `n_estimators` (jointly), `max_depth`, `min_child_weight`/`min_data_in_leaf`, `subsample`/`colsample_bytree`.

## 3. PyTorch

See `model-training.md` for the full training loop pattern. Quick reference:

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
```
Common gotcha: `CrossEntropyLoss` expects raw logits (no softmax applied) and integer class labels, not one-hot.

## 4. TensorFlow / Keras

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
)
```
`sparse_categorical_crossentropy` expects integer labels; `categorical_crossentropy` expects one-hot — a frequent source of shape/label-mismatch bugs.

## 5. Deployment Checklist (light)

- [ ] Serialize the **entire** preprocessing + model pipeline, not just the model — train/serve skew from mismatched preprocessing is a top cause of production failures.
- [ ] Confirm feature computation at serving time matches training time exactly (see feature stores note in `feature-engineering.md`).
- [ ] Decide batch vs. real-time serving based on latency requirements; real-time typically needs a lighter model or optimized runtime (ONNX, TensorRT, quantization).
- [ ] Version the deployed model and keep the ability to roll back.
- [ ] Test with a shadow deployment or canary release before full rollout when possible.
- [ ] Confirm input validation exists in production — production data will eventually violate training-time assumptions (nulls, new categories, out-of-range values).

## 6. Monitoring Checklist (light)

- [ ] Track prediction distribution over time — sudden shifts can indicate data drift even before labels are available to check accuracy.
- [ ] Track feature distributions over time (data drift) separately from prediction drift.
- [ ] Set up delayed ground-truth evaluation where possible (e.g. compare predictions to actual outcomes once labels arrive) to catch concept drift.
- [ ] Alert on missing/null feature values in production that weren't present in training.
- [ ] Log enough to debug a bad prediction after the fact (inputs, model version, prediction, confidence).

For deep infrastructure work (Kubernetes model serving, CI/CD pipelines for models, feature store implementation, A/B testing infrastructure), treat this checklist as a starting point — a dedicated MLOps/infra skill or specialist would go much deeper.
