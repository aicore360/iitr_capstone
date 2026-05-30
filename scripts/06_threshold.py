"""Threshold tuning — pick the F1-optimal operating point.

Usage
-----
    uv run python scripts\\06_threshold.py

Reads
-----
    outputs/models/best_model.joblib

Writes
------
    outputs/figures/09_threshold_tuning.png
    outputs/reports/metrics.json   (adds best_threshold + best_threshold_f1)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import data, threshold   # noqa: E402


def main() -> None:
    _, X_test, _, y_test = data.get_train_test()
    threshold.find_best_threshold(X_test, y_test)


if __name__ == "__main__":
    main()
