"""Centralised configuration: paths, constants, and column metadata.

Every other module in the package imports from this one. To change anything
project-wide (paths, the random seed, which columns are categorical), edit
this file in one place and the entire pipeline picks it up.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Pass this `random_state=` to every estimator, train_test_split, and
# RandomizedSearchCV. Without a fixed seed, different runs produce different
# models — and the report's numbers wouldn't be reproducible.
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Filesystem paths (derived from this file's location)
# ---------------------------------------------------------------------------
# Using __file__ rather than CWD means the package works no matter where
# the user runs it from (project root, an IDE, a CI runner, etc.).
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PATH: Path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
CV_SCORES_PATH: Path = REPORTS_DIR / "cv_scores.json"


# ---------------------------------------------------------------------------
# Dataset column metadata
# ---------------------------------------------------------------------------
# Source of truth for which columns are categorical vs numerical. The
# preprocessing pipeline reads these lists to build its ColumnTransformer,
# so adding a new column means editing one list here.
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

NUMERICAL_COLUMNS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "SeniorCitizen",        # encoded 0/1 in the raw CSV but semantically categorical
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def ensure_dirs() -> None:
    """Create all output directories if they do not already exist.

    matplotlib's savefig fails silently when the target directory is missing,
    which is a common beginner trap. Call this once at the top of any script
    that writes artefacts to outputs/.
    """
    for d in (FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
