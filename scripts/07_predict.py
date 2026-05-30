"""CLI: score a hand-crafted sample customer using the saved model.

Usage
-----
    uv run python scripts\\07_predict.py

This is the "demo" entry point you can show in the 5-minute presentation:
prove the saved Pipeline takes a raw customer dict (no preprocessing
required by the caller) and returns a churn probability + retention
recommendation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import inference   # noqa: E402

# A made-up customer with ALL the red flags: short tenure, fiber-optic with
# high charges, month-to-month contract, no security/support, electronic
# check payment. The model should flag this person as high-risk.
SAMPLE_CUSTOMER = {
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


def main() -> None:
    result = inference.score_one_customer(SAMPLE_CUSTOMER)
    print("Sample customer:")
    for k, v in SAMPLE_CUSTOMER.items():
        print(f"  {k:>20} = {v}")
    print("\nModel verdict:")
    for k, v in result.items():
        print(f"  {k:>20} = {v}")


if __name__ == "__main__":
    main()
