
import pandas as pd
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"

V1_MODEL_FILE = BASE_DIR / "models" / "caf_model.joblib"
V2_MODEL_FILE = BASE_DIR / "models" / "caf_model_v2.joblib"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

v1_package = joblib.load(V1_MODEL_FILE)
v2_package = joblib.load(V2_MODEL_FILE)

v1_features = v1_package["features"]
v2_features = v2_package["features"]

# Three chronological testing periods
periods = [
    ("Test 1", "2020-01-01", "2021-12-31"),
    ("Test 2", "2022-01-01", "2023-12-31"),
    ("Test 3", "2024-01-01", "2025-12-31")
]

models = {
    "V1 Logistic": v1_package,
    "V2 Logistic/RF": v2_package
}

print("=" * 70)
print("CAF CHAMPIONS LEAGUE - WALK-FORWARD VALIDATION")
print("=" * 70)

all_results = []

for test_name, start_date, end_date in periods:

    train_df = df[df["date"] < start_date]

    test_df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ]

    if len(train_df) == 0 or len(test_df) == 0:
        print(f"\nSkipping {test_name}: insufficient data")
        continue

    print(f"\n{test_name}")
    print("-" * 70)
    print(f"Training matches: {len(train_df)}")
    print(f"Testing matches: {len(test_df)}")
    print(f"Testing period: {start_date} to {end_date}")

    y_test = test_df["result"]

    for model_name, package in models.items():

        features = package["features"]

        X_train = train_df[features]
        y_train = train_df["result"]

        X_test = test_df[features]

        # Clone the saved model's pipeline and train from scratch
        model = clone(package["model"])
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced = balanced_accuracy_score(y_test, predictions)

        loss = log_loss(
            y_test,
            probabilities,
            labels=model.classes_
        )

        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced:.4f}")
        print(f"Log Loss: {loss:.4f}")

        all_results.append({
            "period": test_name,
            "model": model_name,
            "matches": len(test_df),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "log_loss": loss
        })

if all_results:

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(results_df.to_string(index=False))

    summary = results_df.groupby("model").agg({
        "accuracy": "mean",
        "balanced_accuracy": "mean",
        "log_loss": "mean"
    })

    print("\nAVERAGE PERFORMANCE")
    print(summary.to_string())

    output_file = BASE_DIR / "training" / "validation_results.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

else:
    print("\nNo validation results were generated.")

print("\nVALIDATION COMPLETED!")
