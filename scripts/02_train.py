"""Train all 5 models with 5-fold CV; save the winner.

Usage
-----
    uv run python scripts\\02_train.py

Output
------
    outputs/models/best_model.joblib   (fitted Pipeline)
    outputs/reports/cv_scores.json     (per-model CV ROC-AUC)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import random
np.random.seed(42)
random.seed(42)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telco_churn import data, train   # noqa: E402


def main() -> None:
    X_train, X_test, y_train, y_test = data.get_train_test()
    print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.4f}")
    print(f"Test  churn rate: {y_test.mean():.4f}")
    train.train_all(X_train, y_train)


if __name__ == "__main__":
    main()
