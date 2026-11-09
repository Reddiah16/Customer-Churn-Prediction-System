"""Train and evaluate customer churn classification models.

The script compares Logistic Regression and Random Forest pipelines, selects
the model with the best holdout F1 score, and saves it to models/.

Run:

    python train.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

FEATURES = ["tenure", "monthly_charges", "total_charges", "support_calls"]
TARGET = "churn"
RANDOM_STATE = 42


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same basic cleaning used during exploration."""
    cleaned = df.copy().drop_duplicates()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(
        cleaned[numeric_columns].median()
    )
    return cleaned


def build_models() -> dict[str, Pipeline]:
    """Create candidate model pipelines with preprocessing included."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate standard classification metrics for a fitted model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def print_results(
    name: str, metrics: dict, y_test: pd.Series, predictions: list[int]
) -> None:
    """Print a readable metrics summary for one model."""
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1 score : {metrics['f1']:.3f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.3f}")
    print("Confusion matrix:")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual: Stayed", "Actual: Churned"],
        columns=["Predicted: Stayed", "Predicted: Churned"],
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Stayed", "Churned"],
        zero_division=0,
    ))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(pd.read_csv(DATA_PATH))
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}

    print(f"Dataset rows : {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows    : {len(X_test)}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print_results(name, metrics, y_test, predictions.tolist())

    # Select by F1 because the dataset is imbalanced and churn detection matters.
    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["roc_auc"]))
    best_model = fitted_models[best_name]

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "target": TARGET,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "selection_metric": "F1 score",
        "metrics": results,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(
        {
            name: {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
            for name, metrics in results.items()
        }
    ).T.round(3)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")

    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Saved comparison to: {REPORTS_DIR / 'model_comparison.csv'}")
    print(
        "\nNote: this dataset has only 20 records, so these scores are "
        "demonstration results, not reliable production estimates."
    )


if __name__ == "__main__":
    main()
