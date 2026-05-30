"""Feature importance: XGBoost built-in + permutation importance.

Two complementary views:

    * XGBoost.feature_importances_ — fast, but reflects how trees were
      *built* (gain on each split). It's measured over the ENCODED features
      (post one-hot), so it shows which "PaymentMethod=Electronic check"
      dummy column mattered, not "PaymentMethod" overall.
    * permutation_importance — slower but more reliable: shuffles each
      ORIGINAL column and measures the drop in test ROC-AUC. We use this
      view for the business inference slide because it talks in raw
      column names.
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from telco_churn import config


def _savefig(name: str) -> None:
    config.ensure_dirs()
    path = config.FIGURES_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)}")


def plot_xgb_feature_importance(pipeline: Pipeline, top_n: int = 15) -> None:
    """Bar chart of XGBoost feature_importances_ over the ENCODED features."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"  Skipping XGB importance (model is {type(model).__name__})")
        return
    preprocess = pipeline.named_steps["preprocess"]
    feat_names = preprocess.get_feature_names_out().tolist()
    df = pd.DataFrame(
        {"feature": feat_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Top {top_n} encoded features — XGBoost importance")
    _savefig("07_xgb_feature_importance")


def plot_permutation_importance(
    pipeline: Pipeline, X_test, y_test, top_n: int = 15
) -> None:
    """Bar chart of permutation importance over the RAW columns.

    We pass the full Pipeline so permutation_importance shuffles raw
    columns and re-applies preprocessing — giving us per-original-column
    importance, which is what the business audience cares about.
    """
    print("  Running permutation importance (10 repeats — takes ~30s)...")
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    df = pd.DataFrame({
        "feature": X_test.columns,
        "mean": result.importances_mean,
        "std":  result.importances_std,
    }).sort_values("mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"][::-1], df["mean"][::-1],
            xerr=df["std"][::-1], color="#DD8452")
    ax.set_xlabel("Drop in ROC-AUC when column is shuffled")
    ax.set_title(f"Top {top_n} raw features — permutation importance")
    _savefig("08_permutation_importance")


def run_importance(X_test, y_test) -> None:
    """Entry point used by scripts/05_importance.py."""
    pipeline: Pipeline = joblib.load(config.MODEL_PATH)
    print(f"Loaded model from: {config.MODEL_PATH.relative_to(config.PROJECT_ROOT)}")
    print("Generating figures:")
    plot_xgb_feature_importance(pipeline)
    plot_permutation_importance(pipeline, X_test, y_test)
