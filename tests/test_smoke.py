"""Smoke test: prove the saved model loads and scores correctly.

Run after scripts/02_train.py has produced outputs/models/best_model.joblib:

    uv run pytest tests\\ -v

Why these tests exist: an end-to-end ML project can pass every internal
metric and still be broken in deployment if the saved Pipeline can't load
or can't score raw inputs. These two tests catch the most common failure
modes in under a second.
"""
from __future__ import annotations

import joblib
import pandas as pd
import pytest

from telco_churn import config


@pytest.fixture(scope="module")
def model():
    if not config.MODEL_PATH.exists():
        pytest.skip(
            f"{config.MODEL_PATH.name} not found — "
            "run `uv run python scripts\\02_train.py` first."
        )
    return joblib.load(config.MODEL_PATH)


def test_model_has_predict_proba(model) -> None:
    """The Pipeline must expose predict_proba (we need probabilities for ROC)."""
    assert hasattr(model, "predict_proba")


def test_score_sample_customer(model) -> None:
    """End-to-end: take a raw customer dict, get a probability in [0, 1].

    No preprocessing on the caller's side — the Pipeline owns it.
    """
    sample = {
        "gender":           "Male",
        "SeniorCitizen":    0,
        "Partner":          "No",
        "Dependents":       "No",
        "tenure":           2,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "Yes",
        "StreamingMovies":  "Yes",
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "MonthlyCharges":   90.0,
        "TotalCharges":     180.0,
    }
    proba = model.predict_proba(pd.DataFrame([sample]))[0, 1]
    assert 0.0 <= proba <= 1.0
