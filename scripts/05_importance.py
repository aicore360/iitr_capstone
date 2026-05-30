"""Feature importance plots — XGBoost built-in + permutation importance.

Usage
-----
    uv run python scripts\\05_importance.py

Reads
-----
    outputs/models/best_model.joblib   (run scripts 02 and 04 first)

Writes
------
    outputs/figures/07_xgb_feature_importance.png
    outputs/figures/08_permutation_importance.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import data, importance   # noqa: E402


def main() -> None:
    _, X_test, _, y_test = data.get_train_test()
    importance.run_importance(X_test, y_test)


if __name__ == "__main__":
    main()
