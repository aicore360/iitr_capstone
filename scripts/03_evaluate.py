"""Evaluate all 5 models on the test set; write metrics + ROC + confusion matrix.

Usage
-----
    uv run python scripts\\03_evaluate.py

Output
------
    outputs/reports/metrics.json
    outputs/figures/05_roc_curves.png
    outputs/figures/06_confusion_matrix.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import data, evaluate   # noqa: E402


def main() -> None:
    X_train, X_test, y_train, y_test = data.get_train_test()
    evaluate.evaluate_all(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
