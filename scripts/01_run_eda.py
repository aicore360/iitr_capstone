"""Generate EDA figures for the Telco Churn capstone.

Run from the project root:

    uv run python scripts\\01_run_eda.py

Output
------
PNG files in outputs/figures/:
    01_target_distribution.png
    02_numerical_distributions.png
    03_categorical_vs_churn.png
    04_correlation_heatmap.png
    04a_monthly_charges_boxplot.png
    04b_churn_rate_by_gender_senior.png
    04c_churn_rate_by_tenure_cohort.png

Plus a fairness-check printout on stdout.

Prerequisites
-------------
The Kaggle CSV must already be at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.
See README.md for the download link.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the `telco_churn` package importable when running this script directly
# (rather than via `python -m`). Adding the project root to sys.path is the
# simplest beginner-friendly approach — no editable install required.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import data, eda  # noqa: E402  (import after sys.path edit)


def main() -> None:
    df = data.load_raw()
    df = data.clean(df)
    eda.run_all_eda(df)


if __name__ == "__main__":
    main()
