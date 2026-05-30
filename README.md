# Telco Customer Churn Prediction

**Capstone project** — Post Graduate Certificate in Applied Data Science & AI, IIT Roorkee (delivered via CloudxLab).

Predicting customer churn for a telecommunications provider using classical machine learning, with a small neural-network comparison.

## Why this matters

For a telecom operator, acquiring a new subscriber typically costs **5–7× more** than retaining an existing one. A model that flags at-risk customers early lets the retention team intervene with targeted offers before churn happens.

This project takes the IBM Telco Customer Churn dataset (7,043 customers, 21 features), builds a reproducible end-to-end ML pipeline, and translates the model's findings into concrete retention actions.

## Dataset

[IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 rows × 21 columns; binary target `Churn` (~26% positive).

The CSV is **not** committed to this repo. Download it yourself:

1. Visit the Kaggle page above.
2. Click **Download**.
3. Unzip and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/raw/`.

## How to reproduce

Prerequisites: **[uv](https://docs.astral.sh/uv/)** (one-line install on Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`).

```powershell
git clone https://github.com/<your-username>/iitr-capstone-telco-churn.git
cd iitr-capstone-telco-churn
uv sync                                           # install pinned dependencies
# ... download the dataset into data/raw/ ...
uv run python scripts\01_run_eda.py               # exploratory plots
uv run python scripts\02_train.py                 # train all 5 models
uv run python scripts\03_evaluate.py              # test-set metrics + ROC
uv run python scripts\04_tune.py                  # tune the best model
uv run python scripts\05_importance.py            # feature importance
uv run python scripts\06_threshold.py             # threshold tuning
uv run python scripts\07_predict.py               # score a sample customer
uv run pytest tests\ -v                           # smoke test
```

## Project structure

```
iitr-capstone/
├── pyproject.toml           uv-managed dependencies
├── data/raw/                Kaggle CSV goes here (gitignored)
├── telco_churn/             main Python package
│   ├── config.py            paths, RANDOM_STATE, column metadata
│   ├── data.py              load + clean
│   ├── eda.py               exploratory plots
│   ├── preprocessing.py     ColumnTransformer factory
│   ├── models.py            5-model registry
│   ├── train.py             5-fold CV + fit best
│   ├── evaluate.py          test metrics, ROC, confusion matrix
│   ├── tune.py              RandomizedSearchCV
│   ├── importance.py        feature + permutation importance
│   ├── threshold.py         operating-point tuning
│   └── inference.py         telecom business actions
├── scripts/                 numbered CLI entry points (01..07)
├── outputs/
│   ├── figures/             PNGs (EDA, ROC, confusion matrix, feature importance, threshold)
│   ├── models/              best_model.joblib (gitignored)
│   └── reports/metrics.json
└── tests/test_smoke.py      sanity test
```

## Approach

- **Classical ML primary**: Logistic Regression, Random Forest, XGBoost.
- **Neural network comparison**: scikit-learn `MLPClassifier` (multi-layer perceptron).
- **Baseline**: `DummyClassifier` to prove "better than guessing".
- **Imbalance handling**: `class_weight='balanced'` (~26% positive — no SMOTE needed).
- **Cross-validation**: 5-fold stratified, scoring on ROC-AUC.
- **Tuning**: `RandomizedSearchCV` on the best classical model only.
- **Threshold tuning**: pick the operating point that maximises F1 (or business cost).

## Results

(Populated after `03_evaluate.py` runs. See `outputs/reports/metrics.json` and the figures in `outputs/figures/`.)

## Author

Rajiv Kumar Gupta — May 2026.
