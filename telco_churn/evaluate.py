"""Test-set evaluation: metrics table, ROC curves, confusion matrix.

This module re-fits every pipeline on the training set so the ROC-curve
overlay can show ALL models — not just the saved winner. The metrics table
and the figures here are what the final presentation cites.

Public API
----------
evaluate_all(X_train, y_train, X_test, y_test)
    Refit, score, plot, write metrics.json.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from telco_churn import config
from telco_churn.models import get_models


def _score(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    """Compute Accuracy / Precision / Recall / F1 / ROC-AUC for one model.

    Why all five metrics:
        * Accuracy on its own is misleading on a 26%-positive dataset (a
          "always predict no churn" classifier scores 74% accuracy).
        * Precision/Recall/F1 tell the business cost story.
        * ROC-AUC is threshold-independent, so it's the right metric for
          comparing models BEFORE we tune the operating threshold.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    }


def _plot_roc_curves(fitted: dict[str, Pipeline], X_test, y_test) -> None:
    """Overlay one ROC curve per model on a single axes."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.50)")
    for name, pipe in fitted.items():
        if name == "Dummy":
            continue   # the dummy has a degenerate ROC; skip for clarity
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — test set")
    ax.legend()

    config.ensure_dirs()
    path = config.FIGURES_DIR / "05_roc_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")


def _plot_confusion_matrix(pipeline: Pipeline, X_test, y_test, name: str) -> None:
    """Heat-map of the test-set confusion matrix for the best model."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Stayed", "Pred: Churned"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: Stayed", "True: Churned"])
    ax.set_title(f"Confusion matrix — {name}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    path = config.FIGURES_DIR / "06_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")


def evaluate_all(X_train, y_train, X_test, y_test) -> dict:
    """Fit every model on full train, evaluate on test, write metrics + figures.

    Also loads and scores the persisted `best_model.joblib`. After
    `04_tune.py` runs, that file holds the tuned XGBoost pipeline; this
    adds a "Saved" row to the comparison so the deck can show the tuned
    test-set numbers (not just the tuned CV number).
    """
    print("Refitting all 5 models on the full training set...")
    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, pipe in get_models().items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics[name] = _score(pipe, X_test, y_test)

    # Also score whatever is currently saved on disk (could be tuned or
    # untuned depending on whether 04_tune.py has run yet).
    if config.MODEL_PATH.exists():
        saved = joblib.load(config.MODEL_PATH)
        inner = type(saved.named_steps["model"]).__name__
        label = f"{inner} (saved)"
        fitted[label] = saved
        metrics[label] = _score(saved, X_test, y_test)
        print(f"  Also scored saved model: {label}")

    table = pd.DataFrame(metrics).T
    table = table[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    table = table.sort_values("roc_auc", ascending=False)
    print("\nTest-set metrics:")
    print(table.round(4).to_string())

    best_name = str(table.index[0])
    print(f"\nBest by test ROC-AUC: {best_name}")

    print("\nFigures:")
    _plot_roc_curves(fitted, X_test, y_test)
    _plot_confusion_matrix(fitted[best_name], X_test, y_test, best_name)

    config.ensure_dirs()
    with config.METRICS_PATH.open("w") as f:
        json.dump({"best_model": best_name, "test_metrics": metrics}, f, indent=2)
    print(f"\nSaved metrics to: {config.METRICS_PATH.relative_to(config.PROJECT_ROOT)}")

    return metrics
