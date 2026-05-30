"""Threshold tuning: pick the operating point on Precision/Recall/F1 curves.

Default decision threshold 0.5 is rarely optimal for an imbalanced problem.
For a churn-retention campaign, missing a real churner (FN) is usually more
costly than wasting a retention offer on a customer who would have stayed
(FP) — but the exact cost ratio is business-specific. Here we report the
F1-optimal threshold and let the reader adjust.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from telco_churn import config


def find_best_threshold(X_test, y_test) -> dict[str, float]:
    """Sweep thresholds 0.05..0.95, plot the curves, persist the F1-optimal.

    Returns
    -------
    dict with keys: 'threshold' (chosen value), 'f1' (its F1 score).
    """
    pipeline: Pipeline = joblib.load(config.MODEL_PATH)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    f1_arr = np.asarray(f1s)
    best_idx = int(np.argmax(f1_arr))
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1_arr[best_idx])
    default_f1 = float(f1_arr[np.argmin(np.abs(thresholds - 0.5))])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", color="#4C72B0")
    ax.plot(thresholds, recalls,    label="Recall",    color="#55A868")
    ax.plot(thresholds, f1s,        label="F1",        color="#DD8452", linewidth=2.5)
    ax.axvline(best_t, color="red",  linestyle="--", alpha=0.7,
               label=f"Best F1 = {best_f1:.3f} @ t={best_t:.2f}")
    ax.axvline(0.5,    color="gray", linestyle=":",  alpha=0.7,
               label=f"Default 0.5 → F1={default_f1:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs threshold")
    ax.legend()

    config.ensure_dirs()
    path = config.FIGURES_DIR / "09_threshold_tuning.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")
    print(f"\nBest F1 = {best_f1:.4f} at threshold = {best_t:.2f}")
    print(f"  vs default 0.5 F1 = {default_f1:.4f}")

    # Persist the chosen threshold so 07_predict.py and the PPT slide can
    # reference it without recomputing.
    try:
        m = json.loads(config.METRICS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        m = {}
    m["best_threshold"] = best_t
    m["best_threshold_f1"] = best_f1
    m["default_threshold_f1"] = default_f1
    config.METRICS_PATH.write_text(json.dumps(m, indent=2))
    print(f"Updated metrics file: {config.METRICS_PATH.relative_to(config.PROJECT_ROOT)}")

    return {"threshold": best_t, "f1": best_f1}
