"""Model registry: 5 sklearn Pipelines we compare on the test set.

The five models cover the full spectrum from "is this better than random?"
up to a neural network:

    Dummy        — predicts the majority class. Sanity baseline.
    LogReg       — linear, interpretable, well-calibrated probabilities.
    RandomForest — non-linear, handles feature interactions, robust.
    XGBoost      — gradient boosting; usually the winner on tabular data.
    MLP          — multi-layer perceptron (a small feed-forward neural
                   network). Required for the course's "Deep Learning"
                   angle without taking on TensorFlow's install pain.

Public API
----------
get_models()       Return a dict of {name: fresh unfitted Pipeline}.
CLASSICAL_MODELS   Names eligible for hyperparameter tuning.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from telco_churn import config
from telco_churn.preprocessing import build_preprocessor

# Why scale_pos_weight ≈ 2.77 for XGBoost:
# The class ratio is negatives:positives ≈ 73:26 ≈ 2.77. XGBoost uses this
# value to upweight the rare class during training — the gradient-boosting
# equivalent of class_weight='balanced'.
_XGB_SCALE_POS_WEIGHT: float = 2.77


def _wrap(estimator) -> Pipeline:
    """Wrap an estimator in a Pipeline with a fresh preprocessor.

    The step name "model" is what RandomizedSearchCV's parameter names
    refer to (e.g. `model__n_estimators`) in `tune.py`.
    """
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def get_models() -> dict[str, Pipeline]:
    """Return all 5 pipelines keyed by short, presentation-friendly names."""
    return {
        "Dummy": _wrap(DummyClassifier(
            strategy="most_frequent",
            random_state=config.RANDOM_STATE,
        )),
        "LogReg": _wrap(LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )),
        "RandomForest": _wrap(RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )),
        "XGBoost": _wrap(XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=_XGB_SCALE_POS_WEIGHT,
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
        "MLP": _wrap(MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=config.RANDOM_STATE,
        )),
    }


CLASSICAL_MODELS: tuple[str, ...] = ("LogReg", "RandomForest", "XGBoost")
"""Names eligible for hyperparameter tuning — excludes the Dummy baseline
and the MLP (we don't tune the NN to keep within the 3-day timeline)."""
