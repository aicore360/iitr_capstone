"""Preprocessing pipeline factory.

The Telco Churn dataset has 3 numerical and 16 categorical columns. We build
a single ColumnTransformer that handles both, then wrap it in a sklearn
Pipeline alongside the estimator. This pattern has three benefits:

1. **No data leakage.** When a Pipeline is passed to `cross_val_score`,
   sklearn re-fits the preprocessor inside every fold using only that
   fold's training data. Doing the preprocessing manually before CV would
   leak test-fold statistics into the training-fold scaling.
2. **Single deployable artefact.** Saving the Pipeline (preprocessor + model)
   to disk via joblib means the final `.joblib` file scores raw inputs
   directly — no separate preprocessor file to ship.
3. **Apples-to-apples model comparison.** Every model gets the same
   preprocessing, so differences in scores reflect the model, not the
   feature engineering.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn import config


def build_preprocessor() -> ColumnTransformer:
    """Return a fresh, unfitted ColumnTransformer.

    Numerical features → StandardScaler (mean=0, std=1).
        Required by LogisticRegression and MLPClassifier (they're scale-
        sensitive). Harmless for tree-based models.

    Categorical features → OneHotEncoder(drop='first', handle_unknown='ignore').
        - `drop='first'` avoids the dummy-variable trap (linear models hate
          perfect collinearity among dummy columns).
        - `handle_unknown='ignore'` makes prediction safe when a category
          value appears at inference time that wasn't seen during training.

    Each call returns a NEW instance, so fitting one doesn't bleed into
    pipelines built elsewhere.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERICAL_COLUMNS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                config.CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",   # any unexpected column is silently dropped
    )
