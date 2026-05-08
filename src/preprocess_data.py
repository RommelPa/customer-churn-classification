from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "telco_customer_churn_clean.csv"
TRAIN_MODELING_PATH = PROCESSED_DATA_DIR / "train_modeling.csv"
VALIDATION_MODELING_PATH = PROCESSED_DATA_DIR / "validation_modeling.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"

ID_COLUMN = "customerID"
TARGET = "Churn"
TARGET_LABEL = "ChurnLabel"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. "
            "Place telco_customer_churn.csv inside data/raw/."
        )

    return pd.read_csv(path)


def clean_churn_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the churn dataset.

    TotalCharges is converted from text to numeric.
    Blank TotalCharges values occur when tenure is 0, so they are set to 0.
    """
    data = data.copy()

    data.columns = [column.strip() for column in data.columns]

    # Standardize text columns
    text_columns = data.select_dtypes(include=["object", "string"]).columns.tolist()

    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()

    # Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    invalid_total_charges = data["TotalCharges"].isna()

    invalid_with_zero_tenure = invalid_total_charges & (data["tenure"] == 0)

    if invalid_total_charges.sum() != invalid_with_zero_tenure.sum():
        problematic_rows = data.loc[
            invalid_total_charges & ~invalid_with_zero_tenure,
            [ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges", TARGET],
        ]

        raise ValueError(
            "Some TotalCharges values are invalid but tenure is not 0. "
            "Manual review is required.\n"
            f"{problematic_rows}"
        )

    data.loc[invalid_with_zero_tenure, "TotalCharges"] = 0

    # Binary target
    data[TARGET_LABEL] = data[TARGET].map({"No": 0, "Yes": 1})

    if data[TARGET_LABEL].isna().any():
        raise ValueError("Unexpected Churn values found. Expected only 'Yes' and 'No'.")

    data[TARGET_LABEL] = data[TARGET_LABEL].astype(int)

    return data


def build_feature_groups(data: pd.DataFrame) -> dict:
    """
    Build feature groups for downstream modeling.
    """
    excluded_columns = [ID_COLUMN, TARGET, TARGET_LABEL]

    feature_data = data.drop(columns=excluded_columns)

    numeric_features = feature_data.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_data.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    feature_groups = {
        "id_column": ID_COLUMN,
        "target": TARGET,
        "target_label": TARGET_LABEL,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    return feature_groups


def create_modeling_splits(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train and validation splits.
    """
    train_data, validation_data = train_test_split(
        data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_LABEL],
    )

    return train_data.reset_index(drop=True), validation_data.reset_index(drop=True)


def save_processed_outputs(
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_groups: dict,
) -> None:
    """
    Save processed datasets and feature groups.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    clean_data.to_csv(CLEAN_DATA_PATH, index=False)
    train_data.to_csv(TRAIN_MODELING_PATH, index=False)
    validation_data.to_csv(VALIDATION_MODELING_PATH, index=False)

    with open(FEATURE_GROUPS_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_groups, file, indent=4)


def print_preprocessing_summary(
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_groups: dict,
) -> None:
    """
    Print preprocessing summary.
    """
    print("=" * 80)
    print("TELCO CHURN PREPROCESSING SUMMARY")
    print("=" * 80)

    print("\n1. Dataset shapes")
    print(f"Clean data: {clean_data.shape[0]:,} rows, {clean_data.shape[1]:,} columns")
    print(f"Train split: {train_data.shape[0]:,} rows, {train_data.shape[1]:,} columns")
    print(
        f"Validation split: {validation_data.shape[0]:,} rows, "
        f"{validation_data.shape[1]:,} columns"
    )

    print("\n2. Target distribution - full clean dataset")
    print(
        clean_data[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n3. Target distribution - train split")
    print(
        train_data[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n4. Target distribution - validation split")
    print(
        validation_data[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n5. TotalCharges validation")
    print(f"Missing TotalCharges after cleaning: {clean_data['TotalCharges'].isna().sum():,}")
    print(clean_data["TotalCharges"].describe())

    print("\n6. Feature groups")
    print(f"Numeric features: {len(feature_groups['numeric_features'])}")
    print(feature_groups["numeric_features"])

    print(f"\nCategorical features: {len(feature_groups['categorical_features'])}")
    print(feature_groups["categorical_features"])

    print("\n7. Files saved")
    print(f"Clean data: {CLEAN_DATA_PATH}")
    print(f"Train modeling data: {TRAIN_MODELING_PATH}")
    print(f"Validation modeling data: {VALIDATION_MODELING_PATH}")
    print(f"Feature groups: {FEATURE_GROUPS_PATH}")


if __name__ == "__main__":
    raw_data = load_raw_data()
    clean_data = clean_churn_data(raw_data)
    feature_groups = build_feature_groups(clean_data)
    train_data, validation_data = create_modeling_splits(clean_data)

    save_processed_outputs(
        clean_data=clean_data,
        train_data=train_data,
        validation_data=validation_data,
        feature_groups=feature_groups,
    )

    print_preprocessing_summary(
        clean_data=clean_data,
        train_data=train_data,
        validation_data=validation_data,
        feature_groups=feature_groups,
    )