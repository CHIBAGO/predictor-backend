
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# CONFIGURATION
# =========================

DATA_DIR = Path("data/bundesliga")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "home_avg_gf5",
    "home_avg_ga5",
    "home_points5",
    "away_avg_gf5",
    "away_avg_ga5",
    "away_points5",
    "home_home_gf5",
    "home_home_ga5",
    "away_away_gf5",
    "away_away_ga5"
]


# =========================
# LOAD DATA
# =========================

def load_data():
    files = sorted(DATA_DIR.glob("D1_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No Bundesliga CSV files found in {DATA_DIR.resolve()}"
        )

    frames = []

    for file in files:
        print("Reading:", file.name)

        df = pd.read_csv(file, encoding="latin-1")

        required = [
            "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR"
        ]

        if not all(col in df.columns for col in required):
            print("Skipped invalid file:", file.name)
            continue

        df = df[required].copy()

        # Parse dates
        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            format="mixed",
            errors="coerce"
        )

        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

        df = df.dropna(
            subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        )

        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        df = df[df["FTR"].isin(["H", "D", "A"])]

        frames.append(df)

    if not frames:
        raise ValueError("No valid Bundesliga data could be loaded.")

    data = pd.concat(frames, ignore_index=True)

    data = data.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"]
    )

    data = data.sort_values("Date").reset_index(drop=True)

    return data


# =========================
# FEATURE ENGINEERING
# =========================

def create_features(data):
    history = {}
    rows = []

    for date, day_matches in data.groupby("Date", sort=True):

        day_rows = []

        # Create features BEFORE updating same-day match results
        for match in day_matches.itertuples(index=False):

            home = match.HomeTeam
            away = match.AwayTeam

            # Ensure every team has all history keys
            h = history.setdefault(
                home,
                {"all": [], "home": [], "away": []}
            )

            a = history.setdefault(
                away,
                {"all": [], "home": [], "away": []}
            )

            hf = h["all"][-5:]
            af = a["all"][-5:]

            hh = h["home"][-5:]
            aa = a["away"][-5:]

            def avg(items, index, default):
                if not items:
                    return default
                return np.mean([x[index] for x in items])

            row = {
                "home_avg_gf5": avg(hf, 0, 1.35),
                "home_avg_ga5": avg(hf, 1, 1.35),
                "home_points5": avg(hf, 2, 1.0),

                "away_avg_gf5": avg(af, 0, 1.10),
                "away_avg_ga5": avg(af, 1, 1.45),
                "away_points5": avg(af, 2, 1.0),

                "home_home_gf5": avg(hh, 0, 1.35),
                "home_home_ga5": avg(hh, 1, 1.35),

                "away_away_gf5": avg(aa, 0, 1.10),
                "away_away_ga5": avg(aa, 1, 1.45),

                "result": match.FTR,
                "date": date
            }

            day_rows.append((row, match))

        # Update team history AFTER features are created
        for row, match in day_rows:

            home = match.HomeTeam
            away = match.AwayTeam

            hg = int(match.FTHG)
            ag = int(match.FTAG)

            hp = 3 if hg > ag else 1 if hg == ag else 0
            ap = 3 if ag > hg else 1 if ag == hg else 0

            # Ensure all keys exist
            h = history.setdefault(
                home,
                {"all": [], "home": [], "away": []}
            )

            a = history.setdefault(
                away,
                {"all": [], "home": [], "away": []}
            )

            # Home team history
            h["all"].append((hg, ag, hp))
            h["home"].append((hg, ag, hp))

            # Away team history
            a["all"].append((ag, hg, ap))
            a["away"].append((ag, hg, ap))

            rows.append(row)

    return pd.DataFrame(rows)


# =========================
# TRAIN MODEL
# =========================

def main():

    print("Loading Bundesliga data...")
    data = load_data()

    print("Matches found:", len(data))

    print("Creating features...")
    table = create_features(data)

    if table.empty:
        raise ValueError("Feature table is empty.")

    X = table[FEATURES]
    y = table["result"]

    # Chronological split
    split = int(len(table) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("Training matches:", len(X_train))
    print("Testing matches:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced"
    )

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n============================")
    print(" BUNDESLIGA MODEL RESULTS")
    print("============================")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    bundle = {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "league": "Bundesliga",
        "league_code": "BL",
        "accuracy": accuracy
    }

    output = MODEL_DIR / "bundesliga_model.joblib"

    joblib.dump(bundle, output)

    print("\nModel saved successfully!")
    print("Model location:", output.resolve())


if __name__ == "__main__":
    main()
