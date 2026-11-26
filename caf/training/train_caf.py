
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "caf_model.joblib"

print("=" * 65)
print("CAF CHAMPIONS LEAGUE - MODEL TRAINING")
print("=" * 65)

# 1. Load data
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Select input features
numeric_features = [
    "home_avg_gf5",
    "home_avg_ga5",
    "home_points5",
    "home_home_gf5",
    "home_home_ga5",
    "home_matches_played",
    "away_avg_gf5",
    "away_avg_ga5",
    "away_points5",
    "away_away_gf5",
    "away_away_ga5",
    "away_matches_played"
]

categorical_features = ["stage"]
feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["result"]

# 3. Chronological train/test split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nTotal matches: {len(df)}")
print(f"Training matches: {len(X_train)}")
print(f"Testing matches: {len(X_test)}")

print(
    f"Training period: {df.iloc[0]['date'].date()} "
    f"to {df.iloc[split_index - 1]['date'].date()}"
)
print(
    f"Testing period: {df.iloc[split_index]['date'].date()} "
    f"to {df.iloc[-1]['date'].date()}"
)

# 4. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42
    )
}

results = {}

# 6. Train and evaluate
for name, classifier in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING: {name}")
    print("=" * 65)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities, labels=pipeline.classes_)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Log Loss: {loss:.4f}")

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

    results[name] = {
        "model": pipeline,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "log_loss": loss
    }

# 7. Majority-class baseline
majority_class = y_train.value_counts().idxmax()
baseline_accuracy = (y_test == majority_class).mean()

print("\n" + "=" * 65)
print("BASELINE COMPARISON")
print("=" * 65)
print(f"Training majority class: {majority_class}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")

# 8. Select model using balanced accuracy
best_name = max(
    results,
    key=lambda name: results[name]["balanced_accuracy"]
)

best_result = results[best_name]

# 9. Save selected model
model_package = {
    "model": best_result["model"],
    "model_name": best_name,
    "features": feature_columns,
    "classes": list(best_result["model"].classes_),
    "metrics": {
        "accuracy": best_result["accuracy"],
        "balanced_accuracy": best_result["balanced_accuracy"],
        "log_loss": best_result["log_loss"]
    }
}

joblib.dump(model_package, MODEL_FILE)

print("\n" + "=" * 65)
print("TRAINING COMPLETED")
print("=" * 65)
print(f"Selected model: {best_name}")
print(f"Saved model: {MODEL_FILE}")
print("SUCCESS!")
