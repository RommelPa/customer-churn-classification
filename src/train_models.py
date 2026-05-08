from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_MODELING_PATH = PROCESSED_DATA_DIR / "train_modeling.csv"
VALIDATION_MODELING_PATH = PROCESSED_DATA_DIR / "validation_modeling.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"

MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load train/validation modeling datasets and feature groups.
    """
    required_files = [
        TRAIN_MODELING_PATH,
        VALIDATION_MODELING_PATH,
        FEATURE_GROUPS_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing processed modeling files. "
            "Run 'python src/preprocess_data.py' first.\n"
            f"Missing files:\n{missing}"
        )

    train_data = pd.read_csv(TRAIN_MODELING_PATH)
    validation_data = pd.read_csv(VALIDATION_MODELING_PATH)

    with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as file:
        feature_groups = json.load(file)

    return train_data, validation_data, feature_groups


def split_features_target(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_groups: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split features and target for train and validation datasets.
    """
    id_column = feature_groups["id_column"]
    target = feature_groups["target"]
    target_label = feature_groups["target_label"]

    drop_columns = [id_column, target, target_label]

    X_train = train_data.drop(columns=drop_columns)
    y_train = train_data[target_label]

    X_valid = validation_data.drop(columns=drop_columns)
    y_valid = validation_data[target_label]

    return X_train, y_train, X_valid, y_valid


def build_preprocessor(feature_groups: dict, scale_numeric: bool) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical features.
    """
    numeric_features = feature_groups["numeric_features"]
    categorical_features = feature_groups["categorical_features"]

    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_models(feature_groups: dict) -> dict[str, Pipeline]:
    """
    Build model pipelines.
    """
    linear_preprocessor = build_preprocessor(feature_groups, scale_numeric=True)
    tree_preprocessor = build_preprocessor(feature_groups, scale_numeric=False)

    models = {
        "baseline_most_frequent": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=30,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def get_positive_class_probabilities(model: Pipeline, X_valid: pd.DataFrame) -> np.ndarray:
    """
    Return predicted probability for the positive class.
    """
    probabilities = model.predict_proba(X_valid)

    return probabilities[:, 1]


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Evaluate a classifier using classification and probability-based metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }

    return metrics


def train_and_evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    """
    Train all models and evaluate them on validation data.
    """
    results = []
    trained_models = {}
    prediction_frames = []

    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")

        pipeline.fit(X_train, y_train)
        y_proba = get_positive_class_probabilities(pipeline, X_valid)

        metrics = evaluate_model(
            model_name=model_name,
            y_true=y_valid,
            y_proba=y_proba,
            threshold=DEFAULT_THRESHOLD,
        )

        results.append(metrics)
        trained_models[model_name] = pipeline

        prediction_frame = pd.DataFrame(
            {
                "model": model_name,
                "actual": y_valid.values,
                "predicted_probability": y_proba,
                "predicted_label": (y_proba >= DEFAULT_THRESHOLD).astype(int),
            }
        )

        prediction_frames.append(prediction_frame)

    results_df = (
        pd.DataFrame(results)
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )

    validation_predictions = pd.concat(
        prediction_frames,
        axis=0,
        ignore_index=True,
    )

    return results_df, trained_models, validation_predictions


def save_outputs(
    results_df: pd.DataFrame,
    trained_models: dict[str, Pipeline],
    validation_predictions: pd.DataFrame,
) -> None:
    """
    Save model metrics, validation predictions, and the best model.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(MODEL_METRICS_PATH, index=False)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    best_model_name = results_df.loc[0, "model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Model metrics: {MODEL_METRICS_PATH}")
    print(f"Validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Best model name: {best_model_name}")


def print_results(results_df: pd.DataFrame) -> None:
    """
    Print model comparison results.
    """
    print("\n" + "=" * 80)
    print("CHURN MODEL COMPARISON RESULTS")
    print("=" * 80)

    formatted_results = results_df.copy()

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    for column in metric_columns:
        formatted_results[column] = formatted_results[column].round(4)

    print(formatted_results)


if __name__ == "__main__":
    train_data, validation_data, feature_groups = load_modeling_data()

    X_train, y_train, X_valid, y_valid = split_features_target(
        train_data=train_data,
        validation_data=validation_data,
        feature_groups=feature_groups,
    )

    model_pipelines = build_models(feature_groups)

    model_results, fitted_models, validation_predictions = train_and_evaluate_models(
        models=model_pipelines,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    print_results(model_results)

    save_outputs(
        results_df=model_results,
        trained_models=fitted_models,
        validation_predictions=validation_predictions,
    )