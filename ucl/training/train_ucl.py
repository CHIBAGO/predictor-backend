"""UEFA Champions League training pipeline for Predictor.

Run from the Predictor project root:
    py ucl/training/train_ucl.py

Reads historical OpenFootball season files and completed 2026/27 results.
Writes only under ucl/. Other league data/models are not modified.
"""
from pathlib import Path
from datetime import datetime
import re
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
UCL_DIR = ROOT / "ucl"
RAW_DIR = UCL_DIR / "data" / "raw"
HIST_DIR = RAW_DIR / "openfootball_historical"
CURRENT_DIR = RAW_DIR / "open_matchday_2627"
PROCESSED_DIR = UCL_DIR / "data" / "processed"
MODEL_DIR = UCL_DIR / "models"
PROCESSED_FILE = PROCESSED_DIR / "ucl_processed.csv"
MODEL_FILE = MODEL_DIR / "ucl_model.joblib"

FEATURES = [
    "home_avg_gf5", "home_avg_ga5", "home_points5",
    "away_avg_gf5", "away_avg_ga5", "away_points5",
    "home_home_gf5", "home_home_ga5",
    "away_away_gf5", "away_away_ga5",
]

# OpenFootball may put the kickoff time only on the first match line
# for a date. Therefore time is optional in this parser.
MATCH_RE = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?"
    r"(.+?)\s+v\s+(.+?)\s+"
    r"(\d+)\s*-\s*(\d+)(?:\s*\((\d+)\s*-\s*(\d+)\))?\s*$"
)

def parse_openfootball_file(path):
    rows = []
    current_date = None
    season = path.parent.name

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "=", "▪", "Group", "Round")):
                continue

            try:
                current_date = pd.Timestamp(
                    datetime.strptime(stripped, "%a %b %d %Y").date()
                )
                continue
            except ValueError:
                pass

            match = MATCH_RE.match(line)
            if not match or current_date is None:
                continue

            _time_text, home, away, hg, ag, _ht_hg, _ht_ag = match.groups()
            home = re.sub(r"\s+\([A-Z]{3}\)$", "", home).strip()
            away = re.sub(r"\s+\([A-Z]{3}\)$", "", away).strip()
            hg, ag = int(hg), int(ag)
            result = "H" if hg > ag else ("A" if hg < ag else "D")

            rows.append({
                "Date": current_date,
                "SeasonCode": season,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": result,
                "Source": "OpenFootball historical",
            })

    return rows

def load_historical():
    files = sorted(HIST_DIR.glob("*/cl.txt"))
    if not files:
        raise FileNotFoundError(
            f"No historical cl.txt files found under {HIST_DIR}. "
            "Run the download script first."
        )

    rows = []
    print(f"Reading {len(files)} historical season files...")
    for path in files:
        season_rows = parse_openfootball_file(path)
        rows.extend(season_rows)
        print(f"  {path.parent.name}: {len(season_rows)} matches")
    return pd.DataFrame(rows)

def load_current_season():
    path = CURRENT_DIR / "results_2627.csv"
    if not path.exists():
        print("WARNING: results_2627.csv not found; current season omitted.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = ["date", "home", "away", "ft_home", "ft_away"]
    if not all(column in df.columns for column in required):
        raise ValueError(
            "Current-season results CSV missing expected columns: "
            + ", ".join(required)
        )

    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df["HomeTeam"] = df["home"].astype(str).str.strip()
    df["AwayTeam"] = df["away"].astype(str).str.strip()
    df["FTHG"] = pd.to_numeric(df["ft_home"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["ft_away"], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG", "HomeTeam", "AwayTeam"])

    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["FTR"] = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"]
        else ("A" if r["FTHG"] < r["FTAG"] else "D"), axis=1
    )
    df["SeasonCode"] = "2627"
    df["Source"] = "Open Matchday current season"
    return df[[
        "Date", "SeasonCode", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "Source"
    ]]

def clean_matches(historical, current):
    frames = [x for x in (historical, current) if not x.empty]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["FTR"] = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"]
        else ("A" if r["FTHG"] < r["FTAG"] else "D"), axis=1
    )
    df = df.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    return df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

def blank_team():
    return {
        "gf": [], "ga": [], "pts": [],
        "home_gf": [], "home_ga": [],
        "away_gf": [], "away_ga": [],
    }

def avg_last(values, default):
    values = values[-5:]
    return float(sum(values) / len(values)) if values else float(default)

def build_features(matches):
    teams, records = {}, []

    for row in matches.itertuples(index=False):
        home, away = str(row.HomeTeam), str(row.AwayTeam)
        h = teams.setdefault(home, blank_team())
        a = teams.setdefault(away, blank_team())

        # Calculate before updating team history: no target leakage.
        records.append({
            "Date": row.Date,
            "SeasonCode": row.SeasonCode,
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
            "Target": row.FTR,
        })

        hg, ag = int(row.FTHG), int(row.FTAG)
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if ag == hg else 0)

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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    historical = load_historical()
    current = load_current_season()
    matches = clean_matches(historical, current)

    if len(matches) < 200:
        raise RuntimeError(
            f"Only {len(matches)} valid matches found. Refusing to train."
        )

    featured = build_features(matches)
    featured.to_csv(PROCESSED_FILE, index=False)

    print("\n========== DATA SUMMARY ==========")
    print(f"Historical rows parsed: {len(historical)}")
    print(f"Current-season completed rows: {len(current)}")
    print(f"Unique valid matches: {len(matches)}")
    print(f"Date range: {matches['Date'].min().date()} to {matches['Date'].max().date()}")
    print(f"Teams: {len(set(matches['HomeTeam']) | set(matches['AwayTeam']))}")
    current_rows = matches[matches["SeasonCode"] == "2627"]
    print(f"2026/27 completed matches included: {len(current_rows)}")
    print("Upcoming fixtures are excluded; only results_2627.csv is read.")

    split = int(len(featured) * 0.80)
    train, test = featured.iloc[:split], featured.iloc[split:]
    if len(train) < 100 or len(test) < 30:
        raise RuntimeError("Not enough data for a safe time-ordered train/test split.")

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

    print("\n========== UCL MODEL EVALUATION ==========")
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Accuracy: {accuracy_score(test['Target'], pred):.4f}")
    try:
        print(f"Log loss: {log_loss(test['Target'], probs, labels=model.classes_):.4f}")
    except Exception as exc:
        print(f"Log loss unavailable: {exc}")
    print(classification_report(test["Target"], pred, zero_division=0))

    bundle = {
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "league": "UEFA Champions League",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "matches_used": int(len(featured)),
        "train_matches": int(len(train)),
        "test_matches": int(len(test)),
        "source": "OpenFootball historical + Open Matchday 2026/27 results",
        "feature_window": 5,
        "season_start": str(matches["SeasonCode"].iloc[0]),
        "season_end": str(matches["SeasonCode"].iloc[-1]),
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nProcessed dataset saved: {PROCESSED_FILE}")
    print(f"UCL model saved: {MODEL_FILE}")
    print("UCL training complete. Other league models were not touched.")

if __name__ == "__main__":
    main()
