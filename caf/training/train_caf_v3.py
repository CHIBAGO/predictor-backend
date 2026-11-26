
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"
OUTPUT_FILE = BASE_DIR / "models" / "caf_model_v3.joblib"

print("=" * 65)
print("CAF CHAMPIONS LEAGUE - FINAL MODEL V3")
print("=" * 65)

# Load dataset
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Load feature list from V2 model package
old_package = joblib.load(
    BASE_DIR / "models" / "caf_model_v2.joblib"
)

features = old_package["features"]

X = df[features]
y = df["result"]

# Identify feature types
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
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

# Best candidate from V3 experiments
classifier = ExtraTreesClassifier(
    n_estimators=500,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# Train using all available historical matches
model.fit(X, y)

# Save model package
package = {
    "model": model,
    "features": features,
    "classes": model.classes_.tolist(),
    "model_name": "Extra Trees Balanced V3",
    "training_matches": len(df),
    "training_start": str(df["date"].min().date()),
    "training_end": str(df["date"].max().date())
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(package, OUTPUT_FILE)

print("\nTRAINING COMPLETED SUCCESSFULLY!")
print("Model:", package["model_name"])
print("Training matches:", len(df))
print("Features:", len(features))
print("Classes:", package["classes"])
print("Saved to:", OUTPUT_FILE)

print("\nCAF V3 MODEL IS READY!")
