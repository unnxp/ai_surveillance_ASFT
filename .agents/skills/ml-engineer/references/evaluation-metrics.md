# Evaluation & Metrics Reference

Covers: metric selection, cross-validation, calibration, statistical significance, over/underfitting diagnosis.

## Table of Contents
1. Metric Selection by Task
2. Cross-Validation Strategy
3. Calibration
4. Statistical Significance of Results
5. Hyperparameter Tuning

---

## 1. Metric Selection by Task

**Binary classification:**
| Situation | Preferred metric(s) | Avoid |
|---|---|---|
| Balanced classes | Accuracy, F1, AUC-ROC | — |
| Imbalanced classes | Precision/Recall, F1, PR-AUC, balanced accuracy | Plain accuracy (misleadingly high) |
| Cost-sensitive (false positives/negatives have different costs) | Custom cost-weighted metric, or optimize threshold on precision/recall for the cheaper error | A single symmetric metric that ignores cost asymmetry |
| Need probability estimates (e.g. ranking, risk scoring) | Log loss, Brier score, calibration curve | Accuracy alone (throws away probability information) |

**Multi-class classification:** macro-F1 (treats classes equally) vs. micro-F1/accuracy (weights by frequency) — pick macro if minority classes matter, micro/weighted if overall correctness matters more.

**Regression:**
| Situation | Preferred metric(s) |
|---|---|
| General purpose | RMSE (penalizes large errors more), MAE (robust to outliers) |
| Relative/percentage error matters | MAPE (careful: undefined/unstable near zero targets) |
| Need a normalized "goodness" score | R² (but can be misleading on non-linear relationships or out-of-distribution test sets) |

**Ranking/recommendation:** NDCG, MAP, MRR — depends on whether position of top results matters (NDCG/MAP) or just the first relevant result (MRR).

**Time series forecasting:** MAE/RMSE on the forecast horizon, plus check errors don't grow unboundedly with horizon length; MASE for comparing across series of different scales.

Always ask: does the offline metric actually correlate with the business outcome the model is meant to improve? A model with better AUC isn't automatically the one to ship if it doesn't move the real target (revenue, click-through, safety).

## 2. Cross-Validation Strategy

- **K-fold CV** — default for i.i.d. data of moderate size; k=5 or k=10 are common defaults.
- **Stratified K-fold** — use for classification to preserve class ratios in each fold, especially with imbalance.
- **Group K-fold** — use when rows share an entity (user, patient) that must not span folds.
- **Time series CV (walk-forward / expanding window)** — never use standard K-fold on temporal data; each fold's training data must precede its validation data.
- **Nested CV** — needed when both hyperparameter tuning and performance estimation happen on the same data; outer loop estimates generalization, inner loop tunes hyperparameters, preventing optimistic bias from tuning on the same data used to report performance.
- Report mean **and** variance (or all fold scores) across CV folds, not just the mean — high variance across folds signals an unstable model or too little data.

## 3. Calibration

- A model can have good discrimination (AUC) but poor calibration (predicted probability of 0.8 doesn't mean 80% actually occurs) — matters whenever predicted probabilities are used directly (risk scores, thresholds set by business stakeholders).
- Check with a reliability diagram / calibration curve (predicted probability bucket vs. observed frequency).
- Fix with **Platt scaling** (logistic regression on top of raw scores) or **isotonic regression** (more flexible, needs more data) — fit the calibrator on a held-out set, not the same data the model trained on.

## 4. Statistical Significance of Results

- A single metric difference between two models (e.g. 0.82 vs 0.83 AUC) may not be meaningful — use paired statistical tests (paired t-test, or a non-parametric alternative like the Wilcoxon signed-rank test) across CV folds or bootstrap resamples to check if the difference is likely real.
- Bootstrap confidence intervals on a metric (resample the test set with replacement many times, recompute the metric) give a practical sense of how much a metric could vary just from test-set sampling noise.
- Be skeptical of leaderboard-chasing on small held-out sets — a small test set has high metric variance, and "improvements" may just be noise.

## 5. Hyperparameter Tuning

- **Grid search** — fine for very small search spaces (few hyperparameters, few values each).
- **Random search** — generally more efficient than grid search for the same compute budget, especially when only a few hyperparameters actually matter.
- **Bayesian optimization** (Optuna, Hyperopt) — more sample-efficient than random search for expensive-to-train models; worth it once each trial takes more than a few minutes.
- Always tune on a validation set (or CV), never on the test set — the test set exists solely to report final, unbiased performance once tuning is done. Tuning on test data is a leakage variant that inflates reported performance.
- Search space should be informed by the algorithm's known sensitivities (e.g. tree depth and learning rate interact strongly in gradient boosting) rather than searching all hyperparameters independently and uniformly.
