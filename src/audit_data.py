from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {path}. "
            "Run the data download/manual placement step first."
        )

    return pd.read_csv(path)


def build_missing_values_report(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build missing values report.
    """
    missing_count = data.isna().sum()
    missing_percent = data.isna().mean() * 100

    missing_report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent.round(2),
            "dtype": data.dtypes.astype(str),
        }
    ).sort_values("missing_percent", ascending=False)

    return missing_report


def audit_total_charges(data: pd.DataFrame) -> None:
    """
    Audit TotalCharges because it is often loaded as text due to blank values.
    """
    print("\n8. TotalCharges audit")

    total_charges_raw = data["TotalCharges"]

    print(f"Raw dtype: {total_charges_raw.dtype}")

    blank_total_charges = total_charges_raw.astype(str).str.strip().eq("").sum()
    print(f"Blank TotalCharges values: {blank_total_charges:,}")

    total_charges_numeric = pd.to_numeric(total_charges_raw, errors="coerce")
    invalid_total_charges = total_charges_numeric.isna().sum()

    print(f"Invalid TotalCharges after numeric conversion: {invalid_total_charges:,}")

    if invalid_total_charges > 0:
        print("\nRows with invalid TotalCharges:")
        print(
            data.loc[
                total_charges_numeric.isna(),
                ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"],
            ].head(20)
        )

    print("\nNumeric TotalCharges summary:")
    print(total_charges_numeric.describe())


def audit_data(data: pd.DataFrame) -> None:
    """
    Print a complete initial audit of the churn dataset.
    """
    print("=" * 80)
    print("TELCO CUSTOMER CHURN DATA AUDIT")
    print("=" * 80)

    print("\n1. Dataset shape")
    print(f"Rows: {data.shape[0]:,}")
    print(f"Columns: {data.shape[1]:,}")

    print("\n2. Column names")
    print(list(data.columns))

    print("\n3. Data types")
    print(data.dtypes)

    print("\n4. Duplicate rows")
    print(f"Duplicate rows: {data.duplicated().sum():,}")

    print("\n5. Duplicate customer IDs")
    print(f"Duplicate customerID values: {data['customerID'].duplicated().sum():,}")

    print("\n6. Missing values report")
    print(build_missing_values_report(data))

    print("\n7. Target distribution")
    target_counts = data["Churn"].value_counts(dropna=False)
    target_percent = data["Churn"].value_counts(normalize=True, dropna=False) * 100

    target_report = pd.DataFrame(
        {
            "count": target_counts,
            "percent": target_percent.round(2),
        }
    )

    print(target_report)

    audit_total_charges(data)

    print("\n9. Numerical features summary")
    numerical_columns = ["SeniorCitizen", "tenure", "MonthlyCharges"]

    for column in numerical_columns:
        print(f"\n{column}")
        print(data[column].describe())

    print("\n10. Categorical cardinality")
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
    categorical_cardinality = (
        data[categorical_columns]
        .nunique(dropna=False)
        .sort_values(ascending=False)
    )

    print(categorical_cardinality)

    print("\n11. Churn rate by important categorical variables")

    important_categorical_columns = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
    ]

    for column in important_categorical_columns:
        churn_rate = (
            data.groupby(column)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .sort_values(ascending=False)
            .round(2)
        )

        print(f"\nChurn rate by {column}")
        print(churn_rate)

    print("\n12. Sample rows")
    print(data.head())


if __name__ == "__main__":
    churn_data = load_raw_data()
    audit_data(churn_data)