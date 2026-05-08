from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from train_models import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "telco_customer_churn_clean.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"

CROSS_VALIDATION_METRICS_PATH = REPORTS_DIR / "cross_validation_metrics.csv"
CROSS_VALIDATION_SUMMARY_PATH = REPORTS_DIR / "cross_validation_summary.csv"

N_SPLITS = 5
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


def load_inputs() -> tuple[pd.DataFrame, dict]:
    """
    Load clean churn data and feature groups.
    """
    required_files = [
        CLEAN_DATA_PATH,
        FEATURE_GROUPS_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing processed files. Run 'python src/preprocess_data.py' first.\n"
            f"Missing files:\n{missing}"
        )

    clean_data = pd.read_csv(CLEAN_DATA_PATH)

    with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as file:
        feature_groups = json.load(file)

    return clean_data, feature_groups


def split_features_target(
    data: pd.DataFrame,
    feature_groups: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split features and binary target.
    """
    id_column = feature_groups["id_column"]
    target = feature_groups["target"]
    target_label = feature_groups["target_label"]

    drop_columns = [id_column, target, target_label]

    X = data.drop(columns=drop_columns)
    y = data[target_label]

    return X, y


def get_positive_class_probabilities(model, X_valid: pd.DataFrame) -> np.ndarray:
    """
    Get predicted probabilities for the positive class.
    """
    probabilities = model.predict_proba(X_valid)
    return probabilities[:, 1]


def evaluate_fold_predictions(
    model_name: str,
    fold: int,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Evaluate one fold using probability and threshold metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "model": model_name,
        "fold": fold,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: dict,
) -> pd.DataFrame:
    """
    Run stratified K-fold cross-validation for all models.
    """
    models = build_models(feature_groups)

    stratified_kfold = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model_pipeline in models.items():
        print(f"Cross-validating {model_name}...")

        for fold_index, (train_index, valid_index) in enumerate(
            stratified_kfold.split(X, y),
            start=1,
        ):
            X_train_fold = X.iloc[train_index]
            X_valid_fold = X.iloc[valid_index]
            y_train_fold = y.iloc[train_index]
            y_valid_fold = y.iloc[valid_index]

            fold_model = clone(model_pipeline)
            fold_model.fit(X_train_fold, y_train_fold)

            y_proba = get_positive_class_probabilities(
                model=fold_model,
                X_valid=X_valid_fold,
            )

            fold_metrics = evaluate_fold_predictions(
                model_name=model_name,
                fold=fold_index,
                y_true=y_valid_fold,
                y_proba=y_proba,
            )

            results.append(fold_metrics)

    return pd.DataFrame(results)


def build_cross_validation_summary(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validation metrics by model.
    """
    summary = (
        cv_results
        .groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
        )
        .sort_values(["roc_auc_mean", "f1_mean"], ascending=False)
        .reset_index(drop=True)
    )

    return summary


def save_cross_validation_outputs(
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> None:
    """
    Save cross-validation tables and figures.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cv_results.to_csv(CROSS_VALIDATION_METRICS_PATH, index=False)
    cv_summary.to_csv(CROSS_VALIDATION_SUMMARY_PATH, index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cv_summary, x="roc_auc_mean", y="model")
    plt.title("Cross-Validation ROC-AUC by Model")
    plt.xlabel("Mean ROC-AUC")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_validation_roc_auc_by_model.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cv_summary, x="f1_mean", y="model")
    plt.title("Cross-Validation F1 by Model")
    plt.xlabel("Mean F1")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_validation_f1_by_model.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cv_summary, x="average_precision_mean", y="model")
    plt.title("Cross-Validation Average Precision by Model")
    plt.xlabel("Mean Average Precision")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_validation_average_precision_by_model.png", dpi=300)
    plt.close()

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Fold metrics: {CROSS_VALIDATION_METRICS_PATH}")
    print(f"Summary metrics: {CROSS_VALIDATION_SUMMARY_PATH}")
    print(f"ROC-AUC figure: {FIGURES_DIR / 'cross_validation_roc_auc_by_model.png'}")
    print(f"F1 figure: {FIGURES_DIR / 'cross_validation_f1_by_model.png'}")
    print(
        "Average precision figure: "
        f"{FIGURES_DIR / 'cross_validation_average_precision_by_model.png'}"
    )


def print_cross_validation_summary(cv_summary: pd.DataFrame) -> None:
    """
    Print cross-validation summary.
    """
    print("\n" + "=" * 80)
    print("CHURN CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    formatted_summary = cv_summary.copy()

    metric_columns = [
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "average_precision_mean",
        "average_precision_std",
    ]

    for column in metric_columns:
        formatted_summary[column] = formatted_summary[column].round(4)

    print(formatted_summary)


if __name__ == "__main__":
    clean_data, feature_groups = load_inputs()
    X, y = split_features_target(clean_data, feature_groups)

    cv_results = run_cross_validation(
        X=X,
        y=y,
        feature_groups=feature_groups,
    )

    cv_summary = build_cross_validation_summary(cv_results)

    print_cross_validation_summary(cv_summary)
    save_cross_validation_outputs(cv_results, cv_summary)