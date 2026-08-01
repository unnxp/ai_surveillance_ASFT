# Data Pipeline Reference

Covers: loading/cleaning data, missing values, outliers, splitting strategy, data leakage, class imbalance.

## Table of Contents
1. Data Loading & Cleaning
2. Missing Values
3. Outliers
4. Train/Val/Test Splitting
5. Data Leakage Checklist
6. Class Imbalance

---

## 1. Data Loading & Cleaning

- Always inspect shape, dtypes, and a sample of rows before doing anything else (`df.info()`, `df.describe()`, `df.head()`).
- Check for duplicate rows explicitly — duplicates that end up split across train/test are a silent leakage source.
- Validate schema assumptions (expected columns, types, ranges) with an assertion or a small validation function rather than assuming the CSV/DB matches expectations.
- Log how many rows/columns were dropped at each cleaning step — silent row loss is a common source of confusing downstream bugs.

```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape, df.dtypes)
n_before = len(df)
df = df.drop_duplicates()
print(f"Dropped {n_before - len(df)} duplicate rows")
```

## 2. Missing Values

- Understand *why* data is missing before choosing a strategy: missing completely at random (MCAR), missing at random (MAR), or missing not at random (MNAR) call for different handling.
- Common strategies, roughly in order of sophistication:
  - Drop rows/columns (only if missingness is small and random, e.g. <5% and MCAR)
  - Simple imputation: mean/median (numeric), mode (categorical)
  - Model-based imputation: KNN imputer, iterative imputer (MICE)
  - Add a missingness indicator column alongside imputation — missingness itself can be predictive
- **Fit imputers on training data only**, then apply to val/test. Fitting on the full dataset before splitting is a leakage bug.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)  # transform only, never fit
```

## 3. Outliers

- Detect via IQR, z-score, or isolation forest depending on dimensionality.
- Decide deliberately whether an outlier is a data error (fix/remove) or a legitimate rare event (keep — especially if the task is to predict rare events, e.g. fraud).
- Tree-based models (XGBoost, random forest) are fairly robust to outliers; linear models and distance-based methods (KNN, k-means) are sensitive — factor the downstream model into the decision.

## 4. Train/Val/Test Splitting

Match the split to how the model will actually be used in production:

| Data type | Correct split strategy | Why |
|---|---|---|
| i.i.d. rows (e.g. one row per independent customer transaction) | Random split | No dependency between rows |
| Time series / any temporal data | Time-based split (train on past, validate on future) | Prevents future information leaking into training |
| Multiple rows per entity (e.g. multiple purchases per user) | Group-based split (`GroupKFold`, split by user ID) | Prevents the same entity appearing in both train and test |
| Small datasets | K-fold or stratified K-fold cross-validation | Single split has too much variance |
| Imbalanced classification | Stratified split | Preserves class ratio in each split |

```python
# Time series — never use a random split
train = df[df["date"] < "2025-01-01"]
val = df[df["date"] >= "2025-01-01"]

# Grouped data — same user must not appear in both sets
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(splitter.split(df, groups=df["user_id"]))
```

## 5. Data Leakage Checklist

Run through this list on every pipeline review — leakage produces models that look great in validation and fail in production.

- [ ] Were scalers/encoders/imputers fit on train only, then applied (`.transform`, not `.fit_transform`) to val/test?
- [ ] For time series: is every training example strictly earlier in time than every validation/test example?
- [ ] For grouped data: does any entity (user, patient, session ID) appear in more than one split?
- [ ] Do any features encode the target directly or indirectly (e.g. a "total_purchases" feature computed *after* the event you're predicting)?
- [ ] Was feature selection or dimensionality reduction (PCA, mutual information ranking) fit on the full dataset before splitting? (Should be fit on train only.)
- [ ] For target encoding of categoricals: was it done with proper cross-validation/out-of-fold encoding, or does it leak the target directly into the training features?
- [ ] Were duplicate/near-duplicate rows removed *before* splitting?
- [ ] If external data was joined in, does the join key's information exist at prediction time in production?

## 6. Class Imbalance

- First check whether imbalance is actually a problem for the chosen metric — e.g., AUC-ROC and average precision are often fine on moderately imbalanced data without resampling.
- Options, not mutually exclusive:
  - Class weights (`class_weight="balanced"` in sklearn, `scale_pos_weight` in XGBoost) — usually the first thing to try, no data duplication needed
  - Oversampling minority class (SMOTE and variants) — apply only on the training fold, never before splitting
  - Undersampling majority class — risk of discarding useful data
  - Threshold tuning at inference time instead of resampling at training time
- Always evaluate with metrics robust to imbalance (precision/recall, F1, PR-AUC) rather than accuracy — see `evaluation-metrics.md`.
