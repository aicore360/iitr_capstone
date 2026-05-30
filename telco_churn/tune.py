"""Hyperparameter tuning for the best classical model via RandomizedSearchCV.

We tune XGBoost (the best classical model on this dataset in practice) with
20 random combinations × 5-fold CV = 100 fits. This finishes in under a
minute on a laptop CPU and typically lifts ROC-AUC by 0.5–1.5 points over
the untuned default.

Why Random search and not Grid search:
    The grid has thousands of combinations. Random search at 20 iterations
    typically lands within 1% of the best grid value in a fraction of the
    time (Bergstra & Bengio, 2012).

Public API
----------
tune_best(X_train, y_train) -> tuned_pipeline
"""
from __future__ import annotations

import json

import joblib
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from telco_churn import config
from telco_churn.models import get_models

# Keys use the `model__` prefix because the estimator inside the Pipeline
# is named "model". e.g. setting "model__max_depth" reaches the XGBClassifier.
XGB_PARAM_DIST: dict = {
    "model__n_estimators":     randint(100, 600),
    "model__max_depth":        randint(3, 9),
    "model__learning_rate":    loguniform(0.01, 0.3),
    "model__subsample":        [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": randint(1, 8),
}


def tune_best(X_train, y_train, n_iter: int = 20) -> Pipeline:
    """Random-search XGBoost hyperparameters and save the winner.

    Replaces the existing `outputs/models/best_model.joblib` so downstream
    scripts (importance, threshold, predict, presentation) pick up the
    tuned pipeline automatically.
    """
    pipe = get_models()["XGBoost"]
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=1,
        refit=True,    # leaves us a fully-fitted pipeline at search.best_estimator_
    )
    print(f"Running RandomizedSearchCV ({n_iter} iters × 5-fold)...")
    search.fit(X_train, y_train)

    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")

    config.ensure_dirs()
    joblib.dump(search.best_estimator_, config.MODEL_PATH)
    print(f"\nSaved tuned model to: {config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}")

    # Append tuning results to cv_scores.json for the report.
    record = {
        "tuned_model": "XGBoost",
        "tuned_cv_roc_auc": float(search.best_score_),
        "best_params": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        },
    }
    try:
        existing = json.loads(config.CV_SCORES_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    existing.update(record)
    config.CV_SCORES_PATH.write_text(json.dumps(existing, indent=2))

    return search.best_estimator_
