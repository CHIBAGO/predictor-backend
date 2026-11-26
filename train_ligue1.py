"""Ligue 1 (France) model training pipeline for Predictor.

Downloads historical French Ligue 1 match results from football-data.co.uk,
builds pre-match rolling team features, evaluates a time-ordered holdout, and
saves models/ligue1_model.joblib.

Run from the Predictor project root:
    py train_ligue1.py

This script only rebuilds the Ligue 1 dataset/model. It does not modify other
league models or their data.
"""
from pathlib import Path
from datetime import datetime
import time
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ligue1_training" / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
PROCESSED_FILE = DATA_DIR / "ligue1_processed.csv"
MODEL_FILE = MODEL_DIR / "ligue1_model.joblib"

# Season codes: 9394 = 1993/94, 2526 = 2025/26.
# Older seasons are included to recover Le Mans' former Ligue 1 matches.
SEASONS = [
    "9394", "9495", "9596", "9697", "9798", "9899",
    "9900", "0001", "0102", "0203", "0304", "0405",
    "0506", "0607", "0708", "0809", "0910",
    "1011", "1112", "1213", "1314", "1415", "1516",
    "1617", "1718", "1819", "1920", "2021", "2122",
    "2223", "2324", "2425", "2526",
]

# football-data.co.uk uses F1 for French Ligue 1.
CSV_URL = "https://www.football-data.co.uk/mmz4281/{season}/F1.csv"

FEATURES = [
    "home_avg_gf5", "home_avg_ga5", "home_points5",
    "away_avg_gf5", "away_avg_ga5", "away_points5",
    "home_home_gf5", "home_home_ga5",
    "away_away_gf5", "away_away_ga5",
]

# Normalize historical spellings to one stable club name.
# Only aliases are included here; no synthetic match records are created.
TEAM_ALIASES = {
    "Le Mans UC 72": "Le Mans",
    "MUC 72": "Le Mans",
    "Le Mans UC": "Le Mans",
    "Paris FC": "Paris FC",
}


def normalize_team_name(value):
    name = str(value).strip()
    return TEAM_ALIASES.get(name, name)


def make_session():
    """Retry temporary connection failures instead of abandoning a season."""
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Predictor football research project"})
    return session


def read_season_csv(path):
    """Read normal files first; fall back for older malformed CSV rows."""
    try:
        return pd.read_csv(path, encoding="latin-1")
    except (pd.errors.ParserError, UnicodeDecodeError):
        print("    Standard CSV parser failed; trying tolerant parser...")
        # Some old source files contain malformed rows with extra delimiters.
        # Skip only rows that cannot be parsed and report the fallback.
        return pd.read_csv(
            path, encoding="latin-1", engine="python", on_bad_lines="skip"
        )


def download_seasons():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    failed = []
    session = make_session()
    print("Downloading French Ligue 1 historical CSV data (F1)...")

    for season in SEASONS:
        path = RAW_DIR / f"F1_{season}.csv"
        try:
            # Download missing/empty files. Retries are handled by the session.
            if not path.exists() or path.stat().st_size < 1000:
                last_error = None
                for attempt in range(1, 4):
                    try:
                        response = session.get(
                            CSV_URL.format(season=season), timeout=(20, 60)
                        )
                        response.raise_for_status()
                        if len(response.content) < 1000:
                            raise ValueError("response too small; source may not exist")
                        path.write_bytes(response.content)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        print(f"  Retry {attempt}/3 for {season}: {exc}")
                        if attempt < 3:
                            time.sleep(3 * attempt)
                if last_error is not None:
                    raise last_error

            df = read_season_csv(path)
            needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
            if not all(col in df.columns for col in needed):
                raise ValueError("required columns missing")

            df = df[needed].copy()
            df["SeasonCode"] = season
            frames.append(df)
            print(f"  OK {season}: {len(df)} source rows")
        except Exception as exc:
            failed.append(season)
            print(f"  FAILED {season}: {exc}")

        time.sleep(0.25)

    if not frames:
        raise RuntimeError("No Ligue 1 seasons downloaded. Check internet/source availability.")

    if failed:
        print("\nSeasons not loaded:", ", ".join(failed))
        print("The model will train only on successfully loaded seasons.")
    else:
        print("\nAll requested seasons loaded successfully.")

    return pd.concat(frames, ignore_index=True)


def parse_matches(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
    df["HomeTeam"] = df["HomeTeam"].map(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].map(normalize_team_name)

    df = df.dropna(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    )
    df = df[df["FTR"].isin(["H", "D", "A"])]
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Remove exact duplicate fixtures if the same season was downloaded twice.
    df = df.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    return df.sort_values(["Date", "SeasonCode"]).reset_index(drop=True)


def _blank():
    return {
        "gf": [], "ga": [], "pts": [],
        "home_gf": [], "home_ga": [],
        "away_gf": [], "away_ga": [],
    }


def _avg_last(values, default):
    values = values[-5:]
    return float(sum(values) / len(values)) if values else float(default)


def build_features(df):
    teams, records = {}, []

    for row in df.itertuples(index=False):
        home, away = str(row.HomeTeam), str(row.AwayTeam)
        h = teams.setdefault(home, _blank())
        a = teams.setdefault(away, _blank())

        # Calculate features before adding this match: avoids target leakage.
        records.append({
            "Date": row.Date,
            "SeasonCode": row.SeasonCode,
            "HomeTeam": home,
            "AwayTeam": away,
            "home_avg_gf5": _avg_last(h["gf"], 1.35),
            "home_avg_ga5": _avg_last(h["ga"], 1.35),
            "home_points5": _avg_last(h["pts"], 1.0),
            "away_avg_gf5": _avg_last(a["gf"], 1.10),
            "away_avg_ga5": _avg_last(a["ga"], 1.45),
            "away_points5": _avg_last(a["pts"], 1.0),
            "home_home_gf5": _avg_last(h["home_gf"], 1.35),
            "home_home_ga5": _avg_last(h["home_ga"], 1.35),
            "away_away_gf5": _avg_last(a["away_gf"], 1.10),
            "away_away_ga5": _avg_last(a["away_ga"], 1.45),
            "Target": row.FTR,
        })

        hg, ag = int(row.FTHG), int(row.FTAG)
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)

        h["gf"].append(hg)
        h["ga"].append(ag)
        h["pts"].append(hp)
        h["home_gf"].append(hg)
        h["home_ga"].append(ag)

        a["gf"].append(ag)
        a["ga"].append(hg)
        a["pts"].append(ap)
        a["away_gf"].append(ag)
        a["away_ga"].append(hg)

    return pd.DataFrame(records)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    matches = parse_matches(download_seasons())
    featured = build_features(matches).sort_values("Date").reset_index(drop=True)

    if len(featured) < 500:
        raise RuntimeError(
            f"Only {len(featured)} valid matches found; refusing to train."
        )

    # Save the complete processed dataset, including the season identifier.
    featured.to_csv(PROCESSED_FILE, index=False)

    # Show whether the intended clubs were actually found in the source data.
    all_teams = set(featured["HomeTeam"]) | set(featured["AwayTeam"])
    print("\n=== Historical team coverage ===")
    for team in ["Le Mans", "Paris FC"]:
        team_rows = featured[
            (featured["HomeTeam"] == team) | (featured["AwayTeam"] == team)
        ]
        if team in all_teams:
            print(
                f"{team}: {len(team_rows)} matches "
                f"({team_rows['Date'].min().date()} to "
                f"{team_rows['Date'].max().date()})"
            )
        else:
            print(f"{team}: NOT FOUND in downloaded Ligue 1 data")

    # Explicitly report whether the latest season was present in the source data.
    latest_rows = featured[featured["SeasonCode"] == "2526"]
    print(f"2025/26 rows loaded: {len(latest_rows)}")
    if latest_rows.empty:
        print("WARNING: 2025/26 data is still missing; Paris FC cannot be learned from this run.")

    split = int(len(featured) * 0.80)
    train, test = featured.iloc[:split], featured.iloc[split:]
    if len(train) < 100 or len(test) < 30:
        raise RuntimeError("Not enough data for a safe temporal train/test split.")

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

    print("\n=== Ligue 1 Model Evaluation (time-ordered holdout) ===")
    print(
        f"Total matches: {len(featured)} | "
        f"Train: {len(train)} | Test: {len(test)}"
    )
    print(f"Date range: {featured['Date'].min().date()} to {featured['Date'].max().date()}")
    print(f"Accuracy: {accuracy_score(test['Target'], pred):.4f}")
    try:
        print(
            f"Log loss: "
            f"{log_loss(test['Target'], probs, labels=model.classes_):.4f}"
        )
    except Exception as exc:
        print(f"Log loss unavailable: {exc}")

    print(classification_report(test["Target"], pred, zero_division=0))

    bundle = {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "league": "Ligue 1",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "matches_used": int(len(featured)),
        "train_matches": int(len(train)),
        "test_matches": int(len(test)),
        "source": "football-data.co.uk French Ligue 1 (F1)",
        "feature_window": 5,
        "season_start": str(featured["SeasonCode"].iloc[0]),
        "season_end": str(featured["SeasonCode"].iloc[-1]),
    }
    joblib.dump(bundle, MODEL_FILE)

    print(f"\nProcessed dataset saved: {PROCESSED_FILE}")
    print(f"Model saved: {MODEL_FILE}")
    print("Ligue 1 training complete. Other league models were not touched.")


if __name__ == "__main__":
    main()
