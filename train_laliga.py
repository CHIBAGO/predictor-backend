"""
LaLiga AI training pipeline for Predictor.
Downloads football-data.co.uk Spanish La Liga CSV seasons (SP1),
builds pre-match rolling team features, trains and evaluates a separate model,
and saves laliga_model.joblib.

Run from the Predictor project root:
    py laliga_training\train_laliga.py
"""

from pathlib import Path
from datetime import datetime
import time
import warnings

import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "laliga_training" / "data"
MODEL_DIR = ROOT / "models"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_FILE = DATA_DIR / "laliga_processed.csv"
MODEL_FILE = MODEL_DIR / "laliga_model.joblib"

# Football-data season code: 2020/21 -> 2021
SEASONS = [
    "1011","1112","1213","1314","1415","1516","1617","1718",
    "1819","1920","2021","2122","2223","2324","2425"
]
CSV_URL = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"

FEATURES = [
    "home_avg_gf5","home_avg_ga5","home_points5",
    "away_avg_gf5","away_avg_ga5","away_points5",
    "home_home_gf5","home_home_ga5","away_away_gf5","away_away_ga5",
]

def download_seasons():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    print("Downloading LaLiga historical CSV data...")
    for season in SEASONS:
        url = CSV_URL.format(season=season)
        path = RAW_DIR / f"SP1_{season}.csv"
        try:
            if not path.exists() or path.stat().st_size < 1000:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                if len(response.content) < 1000:
                    print(f"  Skip {season}: response too small")
                    continue
                path.write_bytes(response.content
                )
            df = pd.read_csv(path, encoding="latin-1")
            needed = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]
            if not all(col in df.columns for col in needed):
                print(f"  Skip {season}: required columns missing")
                continue
            df = df[needed].copy()
            df["SeasonCode"] = season
            frames.append(df)
            print(f"  OK {season}: {len(df)} matches")
        except Exception as e:
            print(f"  Could not download/read {season}: {e}")
        time.sleep(0.15)
    if not frames:
        raise RuntimeError("No seasons downloaded. Check internet or source availability.")
    all_matches = pd.concat(frames, ignore_index=True)
    return all_matches

def parse_dates(df):
    # Football-data CSV date formats can vary across seasons.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG","FTAG"])
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
    df = df[df["FTR"].isin(["H","D","A"])]
    df = df.sort_values(["Date"]).reset_index(drop=True)
    return df

def blank_stats():
    return {"gf": [], "ga": [], "pts": [], "home_gf": [], "home_ga": [], "away_gf": [], "away_ga": []}

def avg_last(values, default):
    values = values[-5:]
    return float(sum(values) / len(values)) if values else float(default)

def build_features(df):
    teams = {}
    records = []
    for _, row in df.iterrows():
        home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
        if home not in teams: teams[home] = blank_stats()
        if away not in teams: teams[away] = blank_stats()
        h, a = teams[home], teams[away]

        # Features are calculated BEFORE updating stats with this match (no target leakage).
        rec = {
            "Date": row["Date"],
            "HomeTeam": home,
            "AwayTeam": away,
            "home_avg_gf5": avg_last(h["gf"], 1.35),
            "home_avg_ga5": avg_last(h["ga"], 1.35),
            "home_points5": avg_last(h["pts"], 1.0),
            "away_avg_gf5": avg_last(a["gf"], 1.10),
            "away_avg_ga5": avg_last(a["ga"], 1.45),
            "away_points5": avg_last(a["pts"], 1.0),
            "home_home_gf5": avg_last(h["home_gf"], 1.35),
            "home_home_ga5": avg_last(h["home_ga"], 1.35),
            "away_away_gf5": avg_last(a["away_gf"], 1.10),
            "away_away_ga5": avg_last(a["away_ga"], 1.45),
            "Target": row["FTR"],
        }
        records.append(rec)

        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        hpts = 3 if hg > ag else (1 if hg == ag else 0)
        apts = 3 if ag > hg else (1 if hg == ag else 0)
        h["gf"].append(hg); h["ga"].append(ag); h["pts"].append(hpts)
        h["home_gf"].append(hg); h["home_ga"].append(ag)
        a["gf"].append(ag); a["ga"].append(hg); a["pts"].append(apts)
        a["away_gf"].append(ag); a["away_ga"].append(hg)
    return pd.DataFrame(records)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = download_seasons()
    matches = parse_dates(raw)
    featured = build_features(matches)
    featured.to_csv(PROCESSED_FILE, index=False)
    print(f"\nProcessed data saved: {PROCESSED_FILE}")
    print(f"Matches available: {len(featured)}")

    # Temporal holdout: the latest 20% of matches are kept for testing.
    featured = featured.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(featured) * 0.80)
    train = featured.iloc[:split_idx]
    test = featured.iloc[split_idx:]
    if len(train) < 100 or len(test) < 30:
        raise RuntimeError("Not enough data to train/test safely.")

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train["Target"])
    pred = model.predict(test[FEATURES])
    probs = model.predict_proba(test[FEATURES])

    print("\n=== LaLiga Model Evaluation (time-ordered holdout) ===")
    print(f"Train matches: {len(train)}")
    print(f"Test matches : {len(test)}")
    print(f"Accuracy     : {accuracy_score(test['Target'], pred):.4f}")
    try:
        print(f"Log loss    : {log_loss(test['Target'], probs, labels=model.classes_):.4f}")
    except Exception as exc:
        print(f"Log loss could not be calculated: {exc}")
    print("\nClassification report:")
    print(classification_report(test["Target"], pred, zero_division=0))

    bundle = {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "league": "LaLiga",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "matches_used": int(len(featured)),
        "train_matches": int(len(train)),
        "test_matches": int(len(test)),
        "source": "football-data.co.uk Spanish La Liga (SP1)",
        "feature_window": 5,
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nModel saved: {MODEL_FILE}")
    print("Training pipeline finished.")

if __name__ == "__main__":
    main()
