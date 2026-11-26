
import pandas as pd
import joblib

from pathlib import Path
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from caf.scripts.predict_features import build_prediction_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "models" / "caf_model_v3.joblib"
DATA_FILE = BASE_DIR / "data" / "processed" / "caf_matches.csv"

RESULT_NAMES = {
    "H": "HOME WIN",
    "D": "DRAW",
    "A": "AWAY WIN"
}


def predict_match(home, away, match_date, stage):

    # Check whether teams exist in historical dataset
    df = pd.read_csv(DATA_FILE)

    known_teams = set(df["home_team"]) | set(df["away_team"])

    if home not in known_teams:
        raise ValueError(f"Home team not found: {home}")

    if away not in known_teams:
        raise ValueError(f"Away team not found: {away}")

    # Generate the 25 features
    features = build_prediction_features(
        home,
        away,
        match_date,
        stage
    )

    # Load trained model
    package = joblib.load(MODEL_FILE)

    model = package["model"]
    feature_names = package["features"]

    # Arrange features in the exact training order
    X = pd.DataFrame([features])[feature_names]

    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    print("\n" + "=" * 55)
    print("CAF CHAMPIONS LEAGUE - MATCH PREDICTION")
    print("=" * 55)

    print(f"Home Team: {home}")
    print(f"Away Team: {away}")
    print(f"Match Date: {match_date}")
    print(f"Stage: {stage}")

    print("\nPREDICTION PROBABILITIES")
    print("-" * 55)

    probability_results = {}

    for class_name, probability in zip(classes, probabilities):
        name = RESULT_NAMES[class_name]
        percentage = probability * 100

        probability_results[class_name] = percentage

        print(f"{name:<15}: {percentage:.2f}%")

    predicted_name = RESULT_NAMES[prediction]

    print("\n" + "=" * 55)
    print(f"MODEL PREDICTION: {predicted_name}")
    print("=" * 55)

    print("\nNOTE:")
    print("These are model estimates, not guaranteed results.")
    print("Prediction quality depends on historical data quality.")

    return {
        "prediction": prediction,
        "probabilities": probability_results
    }


if __name__ == "__main__":

    home = input("Enter home team: ").strip()
    away = input("Enter away team: ").strip()
    match_date = input("Enter match date (YYYY-MM-DD): ").strip()
    stage = input("Enter stage (default Group Stage): ").strip()

    if not stage:
        stage = "Group Stage"

    try:
        pd.to_datetime(match_date, format="%Y-%m-%d", errors="raise")

        predict_match(
            home,
            away,
            match_date,
            stage
        )

    except Exception as error:
        print("\nPREDICTION ERROR:", error)
