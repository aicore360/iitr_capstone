"""Business inference: translate model outputs into telecom retention actions.

The static table maps a top feature to a concrete retention action a
retention manager could take. This is the slide that wins the "top 3"
rank for the CloudxLab blog feature — it shows applied ML, not just
metrics.

Public API
----------
BUSINESS_ACTIONS     List of (feature_pattern, action) for the inference slide.
score_one_customer() Convenience: load saved model, score a customer dict.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from telco_churn import config


# Maps a feature/pattern keyword → recommended retention action.
# Used by 08_build_pptx.py to assemble the Inference slide.
BUSINESS_ACTIONS: list[tuple[str, str]] = [
    (
        "Contract = Month-to-month",
        "Push 12/24-month contract offers with a first-month discount",
    ),
    (
        "InternetService = Fiber optic + high MonthlyCharges",
        "Investigate service-quality / NPS for this segment; cap unexpected price hikes",
    ),
    (
        "OnlineSecurity = No  OR  TechSupport = No",
        "Bundle as a free retention add-on for 6 months",
    ),
    (
        "tenure < 6 months",
        "First-90-days onboarding programme — concierge support call at month 1",
    ),
    (
        "PaymentMethod = Electronic check",
        "Auto-pay migration campaign with one-month bill credit",
    ),
]


def _load_best_threshold(default: float = 0.5) -> float:
    """Read the F1-optimal threshold persisted by threshold.py.

    Falls back to 0.5 if the metrics file is missing — that's deliberate,
    so scoring still works in a fresh checkout that hasn't run the
    threshold-tuning step yet.
    """
    try:
        m = json.loads(config.METRICS_PATH.read_text())
        return float(m.get("best_threshold", default))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def score_one_customer(
    customer: dict, threshold: float | None = None
) -> dict:
    """Score a single customer (dict of raw column names → values).

    Parameters
    ----------
    customer : dict
        Keys must match dataset columns *except* customerID and Churn.
        Values are raw (un-encoded) — the saved Pipeline handles encoding.
    threshold : float, optional
        Decision threshold. Defaults to the F1-optimal value persisted
        in metrics.json (or 0.5 if that file does not yet exist).

    Returns
    -------
    dict with keys:
        churn_probability : float in [0, 1]
        threshold         : float in [0, 1]
        predicted_class   : "Churn" or "Stay"
        recommendation    : business-action string
    """
    pipeline = joblib.load(config.MODEL_PATH)
    df = pd.DataFrame([customer])
    proba = float(pipeline.predict_proba(df)[0, 1])

    if threshold is None:
        threshold = _load_best_threshold()

    will_churn = proba >= threshold
    return {
        "churn_probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "predicted_class": "Churn" if will_churn else "Stay",
        "recommendation": (
            "Trigger retention workflow"
            if will_churn
            else "No action — monitor next quarter"
        ),
    }
