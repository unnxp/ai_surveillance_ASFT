# Feature Engineering Reference

Covers: feature creation/selection, encoding, scaling, dimensionality reduction, feature stores, feature-target leakage.

## Table of Contents
1. Feature Creation
2. Encoding Categorical Variables
3. Scaling & Normalization
4. Feature Selection
5. Dimensionality Reduction
6. Feature-Target Leakage
7. Feature Stores (brief)

---

## 1. Feature Creation

- Start from domain knowledge, not automated feature generation — a handful of well-reasoned features usually beats hundreds of blind combinations.
- Common transforms: ratios, differences, rolling/window aggregates (time series), interaction terms, date-part extraction (day-of-week, is-holiday), text/embedding features.
- For time-based aggregates (e.g. "average purchase in last 30 days"), the aggregate must only use data available *before* the prediction timestamp for that row — this is the most common leakage source in feature engineering.

```python
# Correct: rolling aggregate uses only past data relative to each row
df = df.sort_values("timestamp")
df["rolling_avg_30d"] = (
    df.groupby("user_id")
    .apply(lambda g: g.set_index("timestamp")["amount"].rolling("30D").mean())
    .reset_index(level=0, drop=True)
)
```

## 2. Encoding Categorical Variables

| Method | Best for | Watch out for |
|---|---|---|
| One-hot encoding | Low-cardinality nominal categories | Explodes dimensionality with high cardinality |
| Ordinal encoding | Genuinely ordered categories | Don't use for unordered categories — implies false order |
| Target encoding (mean encoding) | High-cardinality categoricals | **Leaks target info** — must use out-of-fold/cross-validated encoding, add smoothing/regularization |
| Frequency/count encoding | High-cardinality, when frequency is informative | Loses category identity |
| Embeddings (learned) | Very high cardinality, deep learning models | Needs enough data per category to learn well |
| Hashing trick | Very high or unbounded cardinality (e.g. streaming categories) | Hash collisions lose information |

```python
# Correct out-of-fold target encoding (prevents leakage)
from sklearn.model_selection import KFold
import numpy as np

def target_encode_oof(df, col, target, n_splits=5, smoothing=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = np.zeros(len(df))
    global_mean = df[target].mean()
    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        counts = df.iloc[train_idx].groupby(col)[target].count()
        smoothed = (means * counts + global_mean * smoothing) / (counts + smoothing)
        encoded[val_idx] = df.iloc[val_idx][col].map(smoothed).fillna(global_mean)
    return encoded
```

## 3. Scaling & Normalization

- **Fit on train only**, apply to val/test — same rule as imputation.
- Standardization (zero mean, unit variance) — needed for linear models, SVMs, neural nets, PCA, KNN, anything gradient- or distance-based.
- Min-max scaling — useful when a bounded range is required (e.g. some neural net input layers, image pixels).
- Robust scaling (median/IQR-based) — better when outliers are present.
- Tree-based models (random forest, XGBoost, LightGBM) are scale-invariant — scaling is unnecessary for them.

## 4. Feature Selection

- Filter methods (correlation, mutual information, chi-square): fast, model-agnostic, compute on train only.
- Wrapper methods (recursive feature elimination): more expensive, model-specific, prone to overfitting the selection process to the validation set if not cross-validated.
- Embedded methods (L1/Lasso regularization, tree feature importances): selection happens as part of training — generally safest against leakage.
- Always compute correlation/importance/selection statistics on the training fold only, never on the full dataset before splitting.

## 5. Dimensionality Reduction

- PCA: fit on train only, then transform val/test with the same fitted transformer. Standardize features first — PCA is scale-sensitive.
- For high-dimensional sparse data (text, one-hot with high cardinality): consider truncated SVD instead of PCA (handles sparsity better).
- t-SNE/UMAP are for visualization, not for producing features to feed into a downstream supervised model (they don't have a clean `.transform` for new data and can distort distances).

## 6. Feature-Target Leakage

Ask of every feature: "would this value actually be known at the moment of prediction in production?"

Classic examples of leaky features:
- A "days until cancellation" feature when predicting churn (only known in hindsight)
- Aggregates computed over the full dataset (including future rows) instead of only past data relative to the prediction point
- A feature that is a post-hoc label of the outcome (e.g. "was_approved" as a feature when predicting approval)
- IDs or timestamps that happen to correlate with the target for spurious reasons (e.g. row order correlates with time, which correlates with a policy change)

## 7. Feature Stores (brief)

For production systems serving both training and real-time inference, a feature store (e.g. Feast, Tecton, or a homegrown solution) helps guarantee **training/serving consistency** — the same feature computation logic used offline for training is used online for inference, avoiding train/serve skew. Worth recommending when the user describes a production system with a real-time inference path, not needed for one-off model training.
