"""Load and clean the IBM Telco Customer Churn dataset.

Public API
----------
load_raw()           Return the CSV exactly as Kaggle ships it.
clean(df)            Apply minimal, deterministic cleaning.
split(df)            Stratified 80/20 train/test split.
get_train_test()     Convenience wrapper that does all three.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn import config


def load_raw() -> pd.DataFrame:
    """Read the Kaggle CSV from `data/raw/`.

    Returns
    -------
    pd.DataFrame
        Shape (7043, 21) — exactly as Kaggle provides.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing. The error message tells the user where to
        download it.
    """
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at:\n    {config.DATA_PATH}\n\n"
            "Download it from:\n"
            "    https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            f"and place WA_Fn-UseC_-Telco-Customer-Churn.csv into:\n"
            f"    {config.DATA_DIR}\n"
        )
    return pd.read_csv(config.DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Make the dataset model-ready with three deterministic operations.

    1. **Coerce `TotalCharges` to numeric.** The raw column has dtype=object
       because 11 rows contain a single space `" "` instead of a number —
       these are brand-new customers (tenure == 0). After `pd.to_numeric`,
       those rows become NaN; we drop them. 11 / 7043 is < 0.2% of data.
    2. **Map the target.** `Churn` ships as "Yes"/"No"; we map to 1/0 so
       scikit-learn estimators can use it directly.
    3. **Drop `customerID`.** Unique identifiers are never features — they
       carry no generalisable signal and can cause data leakage if left in.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `load_raw()`.

    Returns
    -------
    pd.DataFrame
        Cleaned copy. The input is not mutated.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0})

    df = df.drop(columns=[config.ID_COLUMN])

    return df


def split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split on the cleaned DataFrame.

    Stratification keeps the ~26% positive churn rate in both halves —
    important when the classes are imbalanced, otherwise a small test set
    might end up with very few churners and unstable metrics.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN])
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def get_train_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One-shot helper: load → clean → split. Used by training scripts."""
    df = load_raw()
    df = clean(df)
    return split(df)
