"""Predict whether a customer is likely to churn.

Use the built-in example:

    python predict.py

Or provide customer details:

    python predict.py --tenure 5 --monthly-charges 90 \
        --total-charges 450 --support-calls 5
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FEATURES = ["tenure", "monthly_charges", "total_charges", "support_calls"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict customer churn probability."
    )
    parser.add_argument("--tenure", type=float, default=5, help="Months as a customer")
    parser.add_argument(
        "--monthly-charges", type=float, default=90, help="Monthly bill amount"
    )
    parser.add_argument(
        "--total-charges", type=float, default=450, help="Lifetime charges"
    )
    parser.add_argument(
        "--support-calls", type=float, default=5, help="Number of support calls"
    )
    return parser.parse_args()


def validate_customer(customer: dict[str, float]) -> None:
    """Reject negative values, which cannot describe these features."""
    invalid = [name for name, value in customer.items() if value < 0]
    if invalid:
        raise ValueError(
            "These values cannot be negative: " + ", ".join(invalid)
        )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\nRun 'python train.py' first."
        )

    args = parse_args()
    customer = {
        "tenure": args.tenure,
        "monthly_charges": args.monthly_charges,
        "total_charges": args.total_charges,
        "support_calls": args.support_calls,
    }
    validate_customer(customer)

    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer], columns=FEATURES)

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0][1])

    print("Customer profile:")
    print(customer_df.to_string(index=False))
    print()
    if prediction == 1:
        print("Customer is predicted to churn.")
    else:
        print("Customer is predicted to stay.")
    print(f"Churn probability: {probability:.2%}")


if __name__ == "__main__":
    main()
