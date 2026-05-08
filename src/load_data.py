from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_PATH = RAW_DATA_DIR / "telco_customer_churn.csv"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    if not path.exists():
        available_files = list(RAW_DATA_DIR.glob("*.csv"))

        available = "\n".join(str(file.name) for file in available_files)

        raise FileNotFoundError(
            f"Dataset not found at {path}.\n\n"
            "Download the Telco Customer Churn dataset from Kaggle, "
            "rename the CSV file to 'telco_customer_churn.csv', "
            "and place it inside data/raw/.\n\n"
            f"CSV files currently found in data/raw/:\n{available or 'None'}"
        )

    data = pd.read_csv(path)

    return data


def validate_columns(data: pd.DataFrame) -> None:
    """
    Validate expected dataset columns.
    """
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]

    if missing_columns:
        raise ValueError(
            "The dataset is missing expected columns:\n"
            f"{missing_columns}"
        )


if __name__ == "__main__":
    churn_data = load_raw_data()
    validate_columns(churn_data)

    print("Raw dataset loaded successfully.")
    print(f"Path: {RAW_DATA_PATH}")
    print(f"Rows: {churn_data.shape[0]:,}")
    print(f"Columns: {churn_data.shape[1]:,}")
    print("\nColumn names:")
    print(list(churn_data.columns))

    print("\nTarget distribution:")
    print(churn_data["Churn"].value_counts(dropna=False))

    print("\nTarget distribution percentage:")
    print((churn_data["Churn"].value_counts(normalize=True) * 100).round(2))