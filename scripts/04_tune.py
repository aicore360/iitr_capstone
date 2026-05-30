"""Hyperparameter-tune XGBoost via RandomizedSearchCV (20 iters × 5-fold).

Usage
-----
    uv run python scripts\\04_tune.py

Output
------
    outputs/models/best_model.joblib   (overwritten with tuned XGBoost)
    outputs/reports/cv_scores.json     (appended with best_params + tuned_cv_roc_auc)

After this script, re-run scripts\\03_evaluate.py to see the test-set
improvement from tuning.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import data, tune   # noqa: E402


def main() -> None:
    X_train, X_test, y_train, y_test = data.get_train_test()
    tune.tune_best(X_train, y_train, n_iter=20)


if __name__ == "__main__":
    main()
