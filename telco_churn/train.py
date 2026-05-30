"""5-fold stratified cross-validation, fit the best model on full training.

Why CV instead of a single train/val split:
    A single hold-out validation set gives ONE noisy estimate of model
    quality. 5-fold CV averages 5 estimates, so the ranking between models
    is more reliable — important when picking the model we'll spend the
    rest of the pipeline tuning and explaining.

Public API
----------
train_all(X_train, y_train)
    Run CV for every model, pick the winner, fit it on full train, save
    the pipeline to outputs/models/best_model.joblib, return everything.
"""
from __future__ import annotations

import json

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from telco_churn import config
from telco_churn.models import get_models


def _cv_scores(pipeline: Pipeline, X, y, n_splits: int = 5) -> dict:
    """Return mean / std / per-fold ROC-AUC for one pipeline.

    Note on `n_jobs=1`:
        Windows + multiprocessing + nested estimators (Pipeline inside
        cross_val_score) is a known source of pickle errors. We trade a
        bit of parallelism for reliability; CV finishes in well under a
        minute on this dataset.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=skf, n_jobs=1)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": [float(s) for s in scores],
    }


def train_all(X_train, y_train) -> tuple[str, Pipeline, dict]:
    """Run 5-fold CV on every model, save the winner.

    Returns
    -------
    best_name : str
        Key of the winning model in the registry.
    best_pipeline : Pipeline
        Refitted on the full training set (CV by itself does not leave
        behind a fitted estimator we can use).
    cv_scores : dict[str, dict]
        Per-model {mean, std, folds} ROC-AUC.
    """
    models = get_models()
    print(f"Training {len(models)} models with 5-fold stratified CV...")
    cv_scores: dict[str, dict] = {}
    for name, pipeline in models.items():
        scores = _cv_scores(pipeline, X_train, y_train)
        cv_scores[name] = scores
        print(
            f"  {name:>12}  ROC-AUC = {scores['mean']:.4f}"
            f" (+/- {scores['std']:.4f})"
        )

    # Pick the winner by mean CV ROC-AUC. The Dummy will sit near 0.5 and
    # lose; we still print it as a sanity baseline.
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
    print(f"\nBest by CV ROC-AUC: {best_name} ({cv_scores[best_name]['mean']:.4f})")

    # Build a fresh pipeline (the one we ran CV on hasn't been fitted on
    # the full training set) and fit it on all available training data.
    best_pipeline = get_models()[best_name]
    print(f"Fitting {best_name} on the full training set...")
    best_pipeline.fit(X_train, y_train)

    config.ensure_dirs()
    joblib.dump(best_pipeline, config.MODEL_PATH)
    print(f"Saved best model to: {config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}")

    with config.CV_SCORES_PATH.open("w") as f:
        json.dump({"best_model": best_name, "scores": cv_scores}, f, indent=2)
    print(f"Saved CV scores to:  {config.CV_SCORES_PATH.relative_to(config.PROJECT_ROOT)}")

    return best_name, best_pipeline, cv_scores
