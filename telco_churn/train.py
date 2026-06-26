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
import matplotlib.pyplot as plt
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


def _plot_cv_comparison(cv_scores: dict) -> None:
    """Bar chart of mean CV ROC-AUC per model with error bars."""
    names = list(cv_scores.keys())
    means = [cv_scores[n]["mean"] for n in names]
    stds  = [cv_scores[n]["std"]  for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        names, means, yerr=stds, capsize=5,
        color=["#AAAAAA", "#4C72B0", "#55A868", "#DD8452", "#C44E52"],
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, mean + 0.003,
            f"{mean:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylim(0.45, 0.92)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Random baseline (0.50)")
    ax.set_ylabel("5-fold CV ROC-AUC")
    ax.set_title("Model comparison — 5-fold stratified cross-validation")
    ax.legend()
    plt.tight_layout()
    path = config.FIGURES_DIR / "cv_comparison.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")


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

    _plot_cv_comparison(cv_scores)

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
