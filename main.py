"""Load, clean, and explore the customer churn dataset.

Run this file first to inspect the data and generate exploratory charts:

    python main.py
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script also works on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
REPORTS_DIR = BASE_DIR / "reports"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the customer dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    """Print a first-pass summary of the dataset."""
    print("First five rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and impute missing numeric values."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    numeric_columns = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(
        cleaned[numeric_columns].median()
    )

    print(f"\nData cleaning completed! Rows: {len(cleaned)}")
    print("Missing values after cleaning:")
    print(cleaned.isnull().sum())
    return cleaned


def run_eda(df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> None:
    """Print churn insights and save two exploratory charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("\nChurn counts:")
    print(df["churn"].value_counts().sort_index())

    print("\nAverage monthly charges by churn:")
    print(df.groupby("churn")["monthly_charges"].mean().round(2))

    print("\nAverage feature values by churn:")
    print(
        df.groupby("churn")[["tenure", "monthly_charges", "support_calls"]]
        .mean()
        .round(2)
    )

    print("\nCorrelation with churn:")
    numeric_features = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in {"customer_id", "churn"}
    ]
    correlations = (
        df[numeric_features + ["churn"]]
        .corr(numeric_only=True)["churn"]
        .drop("churn")
        .sort_values(key=abs, ascending=False)
    )
    print(correlations.round(2))

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="churn", hue="churn", palette="Set2", legend=False)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = Stayed, 1 = Left)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    churn_plot = output_dir / "churn_distribution.png"
    plt.savefig(churn_plot, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        df[numeric_features + ["churn"]].corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Feature Correlation")
    plt.tight_layout()
    heatmap_plot = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_plot, dpi=150)
    plt.close()

    print("\nEDA observations:")
    short_tenure = df.loc[df["churn"] == 1, "tenure"].mean()
    long_tenure = df.loc[df["churn"] == 0, "tenure"].mean()
    churn_calls = df.loc[df["churn"] == 1, "support_calls"].mean()
    stay_calls = df.loc[df["churn"] == 0, "support_calls"].mean()
    print(f"- Churned customers average {short_tenure:.1f} months of tenure "
          f"versus {long_tenure:.1f} months for retained customers.")
    print(f"- Churned customers make {churn_calls:.1f} support calls on average "
          f"versus {stay_calls:.1f} for retained customers.")
    print("- Monthly charges also differ across the two churn groups; see the "
          "saved correlation heatmap for the full picture.")
    print(f"\nCharts saved to: {output_dir}")


def main() -> None:
    df = load_data()
    inspect_data(df)
    cleaned_df = clean_data(df)
    run_eda(cleaned_df)


if __name__ == "__main__":
    main()
