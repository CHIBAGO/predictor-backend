
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

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "caf_model_v2.joblib"

print("=" * 65)
print("CAF CHAMPIONS LEAGUE - MODEL TRAINING V2")
print("=" * 65)

# 1. Load V2 dataset
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Select features
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
    "away_matches_played",

    # V2 form features
    "home_win_rate5",
    "away_win_rate5",
    "home_goal_difference5",
    "away_goal_difference5",

    # V2 Elo features
    "home_elo",
    "away_elo",
    "elo_difference",
    "expected_home_score",
    "expected_away_score",

    # V2 head-to-head features
    "h2h_matches",
    "h2h_home_points5",
    "h2h_home_goal_difference5"
]

categorical_features = ["stage"]
feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["result"]

# 3. Same chronological split as V1
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

# 4. Preprocessing function
def create_preprocessor():

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])


# 5. Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42
    )
}

results = {}

# 6. Train and evaluate
for name, classifier in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING V2: {name}")
    print("=" * 65)

    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    loss = log_loss(
        y_test,
        probabilities,
        labels=pipeline.classes_
    )

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

# 7. Baseline
majority_class = y_train.value_counts().idxmax()
baseline_accuracy = (y_test == majority_class).mean()

print("\n" + "=" * 65)
print("V2 BASELINE")
print("=" * 65)
print(f"Majority class: {majority_class}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")

# 8. Compare V1 and V2
print("\n" + "=" * 65)
print("V1 vs V2 COMPARISON")
print("=" * 65)

print("V1 Logistic Regression:")
print("Accuracy: 0.4351")
print("Balanced Accuracy: 0.4266")
print("Log Loss: 1.1339")

print("\nV2 Models:")
for name, result in results.items():
    print(f"\n{name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}")
    print(f"Log Loss: {result['log_loss']:.4f}")

# 9. Select best V2 model by balanced accuracy
best_name = max(
    results,
    key=lambda name: results[name]["balanced_accuracy"]
)

best_result = results[best_name]

# 10. Save V2 separately
model_package = {
    "model": best_result["model"],
    "model_name": best_name,
    "version": "V2",
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
print("V2 TRAINING COMPLETED")
print("=" * 65)
print(f"Selected V2 model: {best_name}")
print(f"Saved model: {MODEL_FILE}")
print("SUCCESS!")
