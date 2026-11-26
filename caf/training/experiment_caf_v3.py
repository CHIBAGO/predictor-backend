
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    log_loss
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"
MODEL_FILE = BASE_DIR / "models" / "caf_model_v2.joblib"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

package = joblib.load(MODEL_FILE)
features = package["features"]

periods = [
    ("Test 1", "2020-01-01", "2021-12-31"),
    ("Test 2", "2022-01-01", "2023-12-31"),
    ("Test 3", "2024-01-01", "2025-12-31")
]

models = {
    "Logistic Balanced": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest Balanced": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees Balanced": ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

print("=" * 75)
print("CAF MODEL V3 - EXPERIMENT")
print("=" * 75)

print(f"Total matches: {len(df)}")
print(f"Features used: {len(features)}")
print("\nFeatures:")
for feature in features:
    print("-", feature)

for period_name, start_date, end_date in periods:

    train_df = df[df["date"] < start_date]
    test_df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ]

    if train_df.empty or test_df.empty:
        continue

    X_train = train_df[features]
    y_train = train_df["result"]

    X_test = test_df[features]
    y_test = test_df["result"]

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_features
        ))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_features
        ))

    preprocessor = ColumnTransformer(transformers)

    print("\n" + "=" * 75)
    print(f"{period_name}: {start_date} to {end_date}")
    print("=" * 75)

    print(f"Training matches: {len(train_df)}")
    print(f"Testing matches: {len(test_df)}")

    majority_class = y_train.value_counts().idxmax()
    baseline = [majority_class] * len(y_test)

    print(
        "Majority baseline accuracy:",
        round(accuracy_score(y_test, baseline), 4)
    )

    for model_name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        accuracy = accuracy_score(y_test, predictions)

        balanced = balanced_accuracy_score(
            y_test,
            predictions
        )

        recalls = recall_score(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            average=None,
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0
        )

        loss = log_loss(
            y_test,
            probabilities,
            labels=pipeline.classes_
        )

        print("\n------------------------------------------")
        print(model_name)
        print("------------------------------------------")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Log loss: {loss:.4f}")

        print(f"Home Win recall: {recalls[0]:.4f}")
        print(f"Draw recall: {recalls[1]:.4f}")
        print(f"Away Win recall: {recalls[2]:.4f}")

        results.append({
            "period": period_name,
            "model": model_name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "log_loss": loss,
            "home_recall": recalls[0],
            "draw_recall": recalls[1],
            "away_recall": recalls[2]
        })

if results:

    results_df = pd.DataFrame(results)

    output_file = (
        BASE_DIR / "training" / "caf_v3_experiment_results.csv"
    )

    results_df.to_csv(output_file, index=False)

    summary = results_df.groupby("model").agg({
        "accuracy": "mean",
        "balanced_accuracy": "mean",
        "macro_f1": "mean",
        "log_loss": "mean",
        "home_recall": "mean",
        "draw_recall": "mean",
        "away_recall": "mean"
    }).sort_values(
        by="balanced_accuracy",
        ascending=False
    )

    print("\n" + "=" * 75)
    print("V3 AVERAGE RESULTS")
    print("=" * 75)

    print(summary.round(4))

    print("\nDetailed results saved to:")
    print(output_file)

print("\nV3 EXPERIMENT COMPLETED!")
