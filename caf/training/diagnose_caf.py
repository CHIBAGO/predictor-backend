
import pandas as pd
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"
V1_FILE = BASE_DIR / "models" / "caf_model.joblib"
V2_FILE = BASE_DIR / "models" / "caf_model_v2.joblib"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

v1_package = joblib.load(V1_FILE)
v2_package = joblib.load(V2_FILE)

periods = [
    ("Test 1", "2020-01-01", "2021-12-31"),
    ("Test 2", "2022-01-01", "2023-12-31"),
    ("Test 3", "2024-01-01", "2025-12-31")
]

models = {
    "V1 Logistic": v1_package,
    "V2 Random Forest": v2_package
}

all_results = []

print("=" * 70)
print("CAF MODEL - DIAGNOSTIC ANALYSIS")
print("=" * 70)

for period_name, start_date, end_date in periods:

    train_df = df[df["date"] < start_date]

    test_df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ]

    if train_df.empty or test_df.empty:
        continue

    y_test = test_df["result"]

    print("\n" + "=" * 70)
    print(f"{period_name}: {start_date} to {end_date}")
    print("=" * 70)

    # Majority-class baseline
    majority_class = train_df["result"].value_counts().idxmax()
    baseline_predictions = [majority_class] * len(test_df)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    print(f"\nTraining majority class: {majority_class}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    for model_name, package in models.items():

        features = package["features"]

        model = clone(package["model"])

        model.fit(
            train_df[features],
            train_df["result"]
        )

        predictions = model.predict(test_df[features])

        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(y_test, predictions)

        print("\n" + "-" * 50)
        print(model_name)
        print("-" * 50)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            target_names=["Home Win", "Draw", "Away Win"],
            zero_division=0
        ))

        print("Confusion Matrix (H, D, A):")
        print(confusion_matrix(
            y_test,
            predictions,
            labels=["H", "D", "A"]
        ))

        all_results.append({
            "period": period_name,
            "model": model_name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced
        })

        # Prediction distribution
        distribution = pd.Series(predictions).value_counts(normalize=True)

        print("\nPrediction distribution:")
        for result in ["H", "D", "A"]:
            print(
                f"{result}: "
                f"{distribution.get(result, 0) * 100:.2f}%"
            )

if all_results:

    results_df = pd.DataFrame(all_results)

    output_file = BASE_DIR / "training" / "diagnostic_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"Results saved to: {output_file}")

print("\nSUCCESS!")
