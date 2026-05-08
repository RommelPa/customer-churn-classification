from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.csv"
THRESHOLD_RECOMMENDATION_PATH = REPORTS_DIR / "threshold_recommendation.csv"

FINAL_MODEL_EVALUATION_PATH = REPORTS_DIR / "final_model_evaluation.csv"
CONFUSION_MATRIX_DEFAULT_PATH = REPORTS_DIR / "confusion_matrix_default_threshold.csv"
CONFUSION_MATRIX_RECOMMENDED_PATH = REPORTS_DIR / "confusion_matrix_recommended_threshold.csv"

SELECTED_MODEL = "gradient_boosting"
DEFAULT_THRESHOLD = 0.50
RECOMMENDED_STRATEGY = "balanced_f1"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load validation predictions and threshold recommendations.
    """
    required_files = [
        VALIDATION_PREDICTIONS_PATH,
        THRESHOLD_RECOMMENDATION_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing evaluation files. Run train_models.py and "
            "threshold_analysis.py first.\n"
            f"Missing files:\n{missing}"
        )

    predictions = pd.read_csv(VALIDATION_PREDICTIONS_PATH)
    threshold_recommendations = pd.read_csv(THRESHOLD_RECOMMENDATION_PATH)

    return predictions, threshold_recommendations


def get_recommended_threshold(threshold_recommendations: pd.DataFrame) -> float:
    """
    Get the selected recommended threshold.
    """
    selected = threshold_recommendations[
        threshold_recommendations["strategy"] == RECOMMENDED_STRATEGY
    ]

    if selected.empty:
        available = threshold_recommendations["strategy"].unique().tolist()
        raise ValueError(
            f"Strategy '{RECOMMENDED_STRATEGY}' not found. "
            f"Available strategies: {available}"
        )

    return float(selected.iloc[0]["threshold"])


def filter_model_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Filter predictions for the selected model.
    """
    model_predictions = predictions[predictions["model"] == SELECTED_MODEL].copy()

    if model_predictions.empty:
        available_models = predictions["model"].unique().tolist()
        raise ValueError(
            f"Model '{SELECTED_MODEL}' not found. Available models: {available_models}"
        )

    return model_predictions


def evaluate_at_threshold(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
    threshold_name: str,
) -> dict:
    """
    Evaluate predictions at a specific threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model": SELECTED_MODEL,
        "threshold_name": threshold_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "customers_flagged": tp + fp,
        "churners_captured": tp,
        "churners_missed": fn,
    }


def build_confusion_matrix_table(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """
    Build a labeled confusion matrix table.
    """
    y_pred = (y_proba >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred)

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )

    return confusion_df


def save_confusion_matrix_plot(
    confusion_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Save a confusion matrix heatmap.
    """
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_probability_curves(
    y_true: pd.Series,
    y_proba: pd.Series,
) -> None:
    """
    Save ROC and Precision-Recall curves.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve — Gradient Boosting")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_gradient_boosting.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision:.3f}")
    plt.title("Precision-Recall Curve — Gradient Boosting")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve_gradient_boosting.png", dpi=300)
    plt.close()


def save_outputs(
    evaluation_summary: pd.DataFrame,
    confusion_default: pd.DataFrame,
    confusion_recommended: pd.DataFrame,
    y_true: pd.Series,
    y_proba: pd.Series,
) -> None:
    """
    Save evaluation outputs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_summary.to_csv(FINAL_MODEL_EVALUATION_PATH, index=False)
    confusion_default.to_csv(CONFUSION_MATRIX_DEFAULT_PATH)
    confusion_recommended.to_csv(CONFUSION_MATRIX_RECOMMENDED_PATH)

    save_confusion_matrix_plot(
        confusion_df=confusion_default,
        output_path=FIGURES_DIR / "confusion_matrix_default_threshold.png",
        title="Confusion Matrix — Default Threshold 0.50",
    )

    save_confusion_matrix_plot(
        confusion_df=confusion_recommended,
        output_path=FIGURES_DIR / "confusion_matrix_recommended_threshold.png",
        title="Confusion Matrix — Recommended Threshold",
    )

    save_probability_curves(y_true=y_true, y_proba=y_proba)

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Final model evaluation: {FINAL_MODEL_EVALUATION_PATH}")
    print(f"Default confusion matrix: {CONFUSION_MATRIX_DEFAULT_PATH}")
    print(f"Recommended confusion matrix: {CONFUSION_MATRIX_RECOMMENDED_PATH}")
    print(f"Default confusion matrix figure: {FIGURES_DIR / 'confusion_matrix_default_threshold.png'}")
    print(f"Recommended confusion matrix figure: {FIGURES_DIR / 'confusion_matrix_recommended_threshold.png'}")
    print(f"ROC curve: {FIGURES_DIR / 'roc_curve_gradient_boosting.png'}")
    print(f"Precision-recall curve: {FIGURES_DIR / 'precision_recall_curve_gradient_boosting.png'}")


def print_evaluation_summary(evaluation_summary: pd.DataFrame) -> None:
    """
    Print final model evaluation summary.
    """
    print("=" * 80)
    print("FINAL CHURN MODEL EVALUATION")
    print("=" * 80)

    display_columns = [
        "threshold_name",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "false_positives",
        "false_negatives",
        "customers_flagged",
        "churners_captured",
        "churners_missed",
    ]

    formatted = evaluation_summary[display_columns].copy()

    for column in [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        formatted[column] = formatted[column].astype(float).round(4)

    print(formatted)


if __name__ == "__main__":
    predictions, threshold_recommendations = load_inputs()

    recommended_threshold = get_recommended_threshold(threshold_recommendations)
    model_predictions = filter_model_predictions(predictions)

    y_true = model_predictions["actual"]
    y_proba = model_predictions["predicted_probability"]

    default_metrics = evaluate_at_threshold(
        y_true=y_true,
        y_proba=y_proba,
        threshold=DEFAULT_THRESHOLD,
        threshold_name="default_0_50",
    )

    recommended_metrics = evaluate_at_threshold(
        y_true=y_true,
        y_proba=y_proba,
        threshold=recommended_threshold,
        threshold_name=RECOMMENDED_STRATEGY,
    )

    evaluation_summary = pd.DataFrame([default_metrics, recommended_metrics])

    confusion_default = build_confusion_matrix_table(
        y_true=y_true,
        y_proba=y_proba,
        threshold=DEFAULT_THRESHOLD,
    )

    confusion_recommended = build_confusion_matrix_table(
        y_true=y_true,
        y_proba=y_proba,
        threshold=recommended_threshold,
    )

    print_evaluation_summary(evaluation_summary)

    save_outputs(
        evaluation_summary=evaluation_summary,
        confusion_default=confusion_default,
        confusion_recommended=confusion_recommended,
        y_true=y_true,
        y_proba=y_proba,
    )