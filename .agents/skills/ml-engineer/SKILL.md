---
name: machine-learning-engineer
description: Comprehensive skill for ML engineering, covering the data-to-model lifecycle with emphasis on data preparation, feature engineering, model development, and training. Use whenever the user asks to write/review ML code (PyTorch, TensorFlow, scikit-learn, XGBoost, etc.), design a data or feature pipeline, clean/preprocess/split datasets, build or debug a training loop, choose/tune a model architecture, evaluate performance, set up experiment tracking, or diagnose overfitting/underfitting/data leakage. Trigger even without the words "machine learning" -- e.g. "help me split this dataset," "why is my model overfitting," "build a recommendation model," "review my training script." Also use for architecture/design tradeoff discussions and concrete engineering advice, not just code.
---

# Machine Learning Engineer

A comprehensive skill for end-to-end ML engineering, with the deepest coverage on **data preparation** and **model development/training** — the two areas where subtle mistakes cause the most damage (leakage, bad splits, silent bugs in training loops, wrong metric choices).

## How to use this skill

1. **Identify where in the lifecycle the request sits** (see Lifecycle Map below) and read the relevant reference file(s) before writing code or giving advice. Don't guess at best practices from memory alone — the references encode specific pitfalls and checklists.
2. **Always state assumptions explicitly** when the user hasn't specified them (task type, data size, framework, latency/compute constraints). Pick sensible defaults and proceed; ask only if the choice would send the whole solution in the wrong direction (e.g., classification vs. regression is unclear and changes everything).
3. **Prefer runnable, complete code** over pseudocode when the user wants an implementation. Save scripts as files per the file-creation rules in the main system prompt (>10 lines of code → create a file).
4. **When reviewing existing code**, actively check it against the pitfall checklists in the references (especially `data-pipeline.md` for leakage and `model-training.md` for training-loop bugs) — don't just check style.

## Lifecycle Map

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐
│ Data         │→ │ Feature       │→ │ Model          │→ │ Evaluation  │→ │ Deployment/ │
│ Preparation  │   │ Engineering   │   │ Development/   │   │ & Tuning    │   │ Monitoring  │
│              │   │               │   │ Training       │   │             │   │ (light)     │
└─────────────┘   └──────────────┘   └───────────────┘   └────────────┘   └────────────┘
      │                   │                   │                  │                │
references/         references/         references/        references/      references/
data-pipeline.md   feature-             model-training.md  evaluation-      frameworks-
                    engineering.md                          metrics.md       cheatsheet.md
                                                                              (deployment
                                                              experiment-      section)
                                                              tracking.md
```

This skill goes deepest on the first three boxes. Deployment/monitoring is covered at a lighter, practical level — for heavy MLOps/infra work (Kubernetes, model servers, CI/CD for models), treat this skill's deployment notes as a starting checklist rather than exhaustive coverage.

## Reference files — read before acting

| File | Read when the task involves... |
|---|---|
| `references/data-pipeline.md` | Loading/cleaning/splitting data, handling missing values or outliers, train/val/test splits, time-series splits, **data leakage** of any kind, class imbalance |
| `references/feature-engineering.md` | Creating/selecting/transforming features, encoding categoricals, scaling, dimensionality reduction, feature stores, feature-target leakage |
| `references/model-training.md` | Writing/reviewing/debugging a training loop, choosing an architecture, regularization, optimizers/schedulers, distributed training, mixed precision, common training bugs |
| `references/evaluation-metrics.md` | Choosing metrics, cross-validation, calibration, statistical significance of results, diagnosing over/underfitting |
| `references/experiment-tracking.md` | Reproducibility, experiment tracking (MLflow/W&B), hyperparameter search, versioning data/models |
| `references/frameworks-cheatsheet.md` | Quick syntax/API lookups across PyTorch, TensorFlow/Keras, scikit-learn, XGBoost/LightGBM, and a short deployment checklist |

Load only the file(s) relevant to the current request — don't read all six for a simple question. For requests spanning multiple stages (e.g. "build me an end-to-end pipeline"), read each relevant file as you reach that stage rather than all upfront.

## Core principles to apply in every response

1. **Leakage first.** Before anything else, check: does any transformation, feature, or split let information from validation/test (or the future) influence training? This is the single most common and most damaging ML engineering mistake. Always fit scalers/encoders/imputers on train only, then transform val/test.
2. **Match the validation strategy to the data.** Random splits are wrong for time series, grouped data (e.g. multiple rows per user), or anything with duplicates/near-duplicates. Ask "what will this model see in production?" and mirror that in the split.
3. **Match the metric to the business/task, not convenience.** Accuracy is usually the wrong metric for imbalanced classification. State *why* a metric fits before using it.
4. **Baseline before complexity.** Recommend a simple baseline (mean/majority-class predictor, logistic regression, gradient-boosted trees) before jumping to deep learning, unless the task clearly requires it (images, text, audio, very large tabular data with complex interactions).
5. **Reproducibility by default.** Set and mention random seeds, log library versions, and note what would need to be tracked (data version, hyperparameters, code version) for the result to be reproducible.
6. **Be explicit about compute/scale assumptions.** State whether code assumes CPU/single-GPU/multi-GPU, and roughly what data size it's designed for, so the user can flag a mismatch.
7. **Surface known failure modes unprompted.** If a design choice has a well-known pitfall (e.g. batch norm with very small batch sizes, target encoding without regularization, LSTM vanishing gradients on long sequences), mention it briefly even if not asked — this is the kind of thing that makes advice from an experienced engineer valuable.

## Output conventions

- **Code**: complete, runnable, with imports; include a short comment on assumptions (framework version, data shape) at the top. Use type hints in Python where reasonable.
- **Architecture/design advice**: give a recommendation with tradeoffs, not just a menu of options — the user came for an engineering opinion. Note alternatives briefly.
- **Debugging requests**: ask for or infer the symptom (loss curve behavior, error message, train vs. val gap) before prescribing a fix; if the user hasn't shared enough to diagnose, give the 2-3 most likely causes ranked by likelihood rather than a generic checklist.
- **Long-form deliverables** (full pipeline scripts, reports comparing models): create files per the standard file-creation rules; keep conversational answers inline.
