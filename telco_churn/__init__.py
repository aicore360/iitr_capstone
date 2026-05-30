"""Telco Customer Churn capstone — main Python package.

Modules
-------
config         : centralised constants (paths, RANDOM_STATE, column metadata).
data           : load and clean the Kaggle CSV.
eda            : exploratory data analysis plots.
preprocessing  : ColumnTransformer factory (scaling + one-hot encoding).
models         : 5-model registry (Dummy / LogReg / RF / XGB / MLP).
train          : 5-fold stratified CV, fit best on full train, save model.
evaluate       : test-set metrics, ROC curves, confusion matrix.
tune           : RandomizedSearchCV on the best classical model.
importance     : feature_importances_ and permutation_importance plots.
threshold      : pick the operating point on Precision/Recall/F1 curves.
inference      : translate top features into telecom retention actions.

The entry points live in scripts/01_run_eda.py .. scripts/08_build_pptx.py
and import from this package.
"""

__version__: str = "0.1.0"
