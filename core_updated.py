
"""Predictor shared core: EPL, LaLiga, Bundesliga and Ligue 1 engines."""

import math
import sqlite3
from pathlib import Path
from functools import lru_cache

import pandas as pd
import requests

DB_FILE = "predictor.db"
NEWCOMER_ELO = 1425.0
HOME_ADVANTAGE = 65.0
K_FACTOR = 20.0
FORM_MATCHES = 5
AVG_GOALS = 1.4
AVG_POINTS = 1.4

FEATURES = [
    "elo_diff", "h_ppg", "h_gf", "h_ga",
    "a_ppg", "a_gf", "a_ga",
    "h_home_ppg", "a_away_ppg", "h_rest", "a_rest"
]

CLASS_NAMES = {"H": "home", "D": "draw", "A": "away"}


# =========================================================
# GENERAL DATABASE
# =========================================================

def load_matches(db_file=DB_FILE):
    with sqlite3.connect(db_file) as con:
        df = pd.read_sql_query(
            "SELECT date, season, home_team, away_team, home_goals, away_goals, result "
            "FROM matches ORDER BY date",
            con
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_goals", "away_goals", "result"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    return df.reset_index(drop=True)


def _push(store, key, value):
    items = store.setdefault(key, [])
    items.append(value)
    del items[:-FORM_MATCHES]


def _average(items, position, default):
    if not items:
        return default
    return sum(x[position] for x in items) / len(items)


# =========================================================
# EPL ENGINE - ORIGINAL
# =========================================================

class TeamState:

    def __init__(self):
        self.elo = {}
        self.recent = {}
        self.home = {}
        self.away = {}
        self.last_date = {}
        self.played = {}
        self.season = None
        self.season_teams = set()

    def get_elo(self, team):
        return self.elo.get(team, NEWCOMER_ELO)

    def new_season(self, season):
        if (
            self.season is not None
            and season != self.season
            and self.season_teams
        ):
            mean = sum(
                self.elo[t] for t in self.season_teams
            ) / len(self.season_teams)

            for team in self.elo:
                self.elo[team] = (
                    0.75 * self.elo[team] + 0.25 * mean
                )

            self.season_teams = set()

        self.season = season

    def known(self, team):
        return self.played.get(team, 0) > 0

    def current_ratings(self):
        teams = self.season_teams or set(self.elo)

        return sorted(
            ((t, self.elo[t]) for t in teams),
            key=lambda x: -x[1]
        )

    def features(self, home, away, date):
        h = self.recent.get(home, [])
        a = self.recent.get(away, [])

        hh = self.home.get(home, [])
        aa = self.away.get(away, [])

        def rest(team):
            last = self.last_date.get(team)
            return 14 if last is None else min(
                14, (date - last).days
            )

        return {
            "elo_diff": self.get_elo(home) - self.get_elo(away),

            "h_ppg": _average(h, 0, AVG_POINTS),
            "h_gf": _average(h, 1, AVG_GOALS),
            "h_ga": _average(h, 2, AVG_GOALS),

            "a_ppg": _average(a, 0, AVG_POINTS),
            "a_gf": _average(a, 1, AVG_GOALS),
            "a_ga": _average(a, 2, AVG_GOALS),

            "h_home_ppg": sum(hh) / len(hh) if hh else AVG_POINTS,
            "a_away_ppg": sum(aa) / len(aa) if aa else AVG_POINTS,

            "h_rest": rest(home),
            "a_rest": rest(away),
        }

    def update(self, date, home, away, home_goals, away_goals):
        eh = self.get_elo(home)
        ea = self.get_elo(away)

        expected_home = 1 / (
            1 + 10 ** (-(eh + HOME_ADVANTAGE - ea) / 400)
        )

        actual_home = (
            1.0 if home_goals > away_goals
            else 0.5 if home_goals == away_goals
            else 0.0
        )

        margin = math.log(abs(home_goals - away_goals) + 1) + 1
        delta = K_FACTOR * margin * (actual_home - expected_home)

        self.elo[home] = eh + delta
        self.elo[away] = ea - delta

        points_home = (
            3 if home_goals > away_goals
            else 1 if home_goals == away_goals
            else 0
        )

        points_away = (
            3 if away_goals > home_goals
            else 1 if away_goals == home_goals
            else 0
        )

        _push(self.recent, home, (points_home, home_goals, away_goals))
        _push(self.recent, away, (points_away, away_goals, home_goals))

        _push(self.home, home, points_home)
        _push(self.away, away, points_away)

        self.last_date[home] = date
        self.last_date[away] = date

        self.played[home] = self.played.get(home, 0) + 1
        self.played[away] = self.played.get(away, 0) + 1

        self.season_teams.update([home, away])


def build_training_table(matches):
    state, rows = TeamState(), []

    for date, day_matches in matches.groupby("date", sort=True):
        day_rows = []

        for m in day_matches.itertuples(index=False):
            state.new_season(m.season)

            row = state.features(
                m.home_team, m.away_team, m.date
            )

            row.update({
                "date": m.date,
                "season": m.season,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "result": m.result,
                "h_played": state.played.get(m.home_team, 0),
                "a_played": state.played.get(m.away_team, 0),
            })

            day_rows.append((row, m))

        for row, m in day_rows:
            state.update(
                m.date,
                m.home_team,
                m.away_team,
                m.home_goals,
                m.away_goals
            )
            rows.append(row)

    return pd.DataFrame(rows), state


def predict_match(bundle, home, away, date):
    state = bundle["state"]

    if not (state.known(home) and state.known(away)):
        return None

    row = pd.DataFrame([
        state.features(home, away, pd.Timestamp(date))
    ])[bundle["features"]]

    probabilities = bundle["model"].predict_proba(row)[0]

    return {
        CLASS_NAMES[c]: float(p)
        for c, p in zip(bundle["model"].classes_, probabilities)
    }


# =========================================================
# LALIGA ENGINE
# =========================================================

LALIGA_FEATURES = [
    "home_avg_gf5", "home_avg_ga5", "home_points5",
    "away_avg_gf5", "away_avg_ga5", "away_points5",
    "home_home_gf5", "home_home_ga5",
    "away_away_gf5", "away_away_ga5"
]

LALIGA_DATA_DIR = Path(__file__).resolve().parent / "data" / "laliga"
LALIGA_DATA_DIR.mkdir(parents=True, exist_ok=True)

LALIGA_SEASONS = [
    f"{y:02d}{(y+1)%100:02d}" for y in range(10, 25)
]


@lru_cache(maxsize=1)
def load_laliga_history():
    frames = []

    for season in LALIGA_SEASONS:
        file_path = LALIGA_DATA_DIR / f"SP1_{season}.csv"

        if not file_path.exists():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"

            try:
                response = requests.get(url, timeout=25)
                response.raise_for_status()
                file_path.write_bytes(response.content)
            except Exception:
                continue

        try:
            df = pd.read_csv(file_path, encoding="latin-1")

            required = [
                "Date", "HomeTeam", "AwayTeam",
                "FTHG", "FTAG", "FTR"
            ]

            if not all(c in df.columns for c in required):
                continue

            df = df[required].dropna().copy()
            df["date"] = pd.to_datetime(
                df["Date"], dayfirst=True, errors="coerce"
            )
            df = df.dropna(subset=["date"])

            df.rename(columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": "result"
            }, inplace=True)

            df["home_goals"] = pd.to_numeric(
                df["home_goals"], errors="coerce"
            )
            df["away_goals"] = pd.to_numeric(
                df["away_goals"], errors="coerce"
            )

            df = df.dropna(
                subset=["home_goals", "away_goals", "result"]
            )

            df["home_goals"] = df["home_goals"].astype(int)
            df["away_goals"] = df["away_goals"].astype(int)

            frames.append(df[[
                "date", "home_team", "away_team",
                "home_goals", "away_goals", "result"
            ]])

        except Exception:
            continue

    if not frames:
        raise RuntimeError(
            "LaLiga historical data unavailable. Check internet connection."
        )

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def laliga_features(history, home, away, cutoff):
    prior = history[
        history["date"] < pd.Timestamp(cutoff)
    ].sort_values("date")

    records = {}

    for row in prior.itertuples(index=False):
        hg, ag = int(row.home_goals), int(row.away_goals)

        hp = 3 if hg > ag else 1 if hg == ag else 0
        ap = 3 if ag > hg else 1 if hg == ag else 0

        rec_h = records.setdefault(
            row.home_team, {"all": [], "home": [], "away": []}
        )

        rec_a = records.setdefault(
            row.away_team, {"all": [], "home": [], "away": []}
        )

        rec_h["all"].append((hg, ag, hp))
        rec_h["home"].append((hg, ag, hp))

        rec_a["all"].append((ag, hg, ap))
        rec_a["away"].append((ag, hg, ap))

    hrec = records.get(
        home, {"all": [], "home": [], "away": []}
    )

    arec = records.get(
        away, {"all": [], "home": [], "away": []}
    )

    hf, af = hrec["all"][-5:], arec["all"][-5:]
    hh, aa = hrec["home"][-5:], arec["away"][-5:]

    def avg(items, idx, default):
        return (
            sum(x[idx] for x in items) / len(items)
            if items else default
        )

    return {
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
    }


def predict_laliga_match(bundle, home, away, match_date):
    history = load_laliga_history()

    if not (
        (history.home_team == home).any()
        or (history.away_team == home).any()
    ):
        return None

    if not (
        (history.home_team == away).any()
        or (history.away_team == away).any()
    ):
        return None

    row = laliga_features(history, home, away, match_date)

    feature_names = bundle.get("features", LALIGA_FEATURES)
    x = pd.DataFrame([row])[feature_names]

    model = bundle["model"]
    probabilities = model.predict_proba(x)[0]

    result = {}

    for cls, probability in zip(
        model.classes_, probabilities
    ):
        result[CLASS_NAMES.get(cls, str(cls).lower())] = float(
            probability
        )

    return {
        k: result.get(k, 0.0)
        for k in ("home", "draw", "away")
    }


def laliga_teams():
    history = load_laliga_history()

    return sorted(
        set(history.home_team.dropna())
        | set(history.away_team.dropna())
    )


# =========================================================
# BUNDESLIGA ENGINE
# =========================================================

BUNDESLIGA_FEATURES = [
    "home_avg_gf5", "home_avg_ga5", "home_points5",
    "away_avg_gf5", "away_avg_ga5", "away_points5",
    "home_home_gf5", "home_home_ga5",
    "away_away_gf5", "away_away_ga5"
]

BUNDESLIGA_DATA_DIR = (
    Path(__file__).resolve().parent / "data" / "bundesliga"
)

BUNDESLIGA_DATA_DIR.mkdir(parents=True, exist_ok=True)

BUNDESLIGA_SEASONS = [
    f"{y:02d}{(y+1)%100:02d}" for y in range(10, 25)
]


@lru_cache(maxsize=1)
def load_bundesliga_history():
    """Load historical Bundesliga match data."""

    frames = []

    for season in BUNDESLIGA_SEASONS:
        file_path = BUNDESLIGA_DATA_DIR / f"D1_{season}.csv"

        if not file_path.exists():
            url = (
                f"https://www.football-data.co.uk/mmz4281/"
                f"{season}/D1.csv"
            )

            try:
                response = requests.get(url, timeout=25)
                response.raise_for_status()
                file_path.write_bytes(response.content)
            except Exception:
                continue

        try:
            df = pd.read_csv(file_path, encoding="latin-1")

            required = [
                "Date", "HomeTeam", "AwayTeam",
                "FTHG", "FTAG", "FTR"
            ]

            if not all(c in df.columns for c in required):
                continue

            df = df[required].dropna().copy()

            df["date"] = pd.to_datetime(
                df["Date"], dayfirst=True, errors="coerce"
            )

            df = df.dropna(subset=["date"])

            df.rename(columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": "result"
            }, inplace=True)

            df["home_goals"] = pd.to_numeric(
                df["home_goals"], errors="coerce"
            )

            df["away_goals"] = pd.to_numeric(
                df["away_goals"], errors="coerce"
            )

            df = df.dropna(
                subset=["home_goals", "away_goals", "result"]
            )

            df["home_goals"] = df["home_goals"].astype(int)
            df["away_goals"] = df["away_goals"].astype(int)

            frames.append(df[[
                "date", "home_team", "away_team",
                "home_goals", "away_goals", "result"
            ]])

        except Exception:
            continue

    if not frames:
        raise RuntimeError(
            "Bundesliga history unavailable. "
            "Check data/bundesliga files or internet connection."
        )

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def bundesliga_features(history, home, away, cutoff):
    """Generate the same rolling features used by Bundesliga training."""

    prior = history[
        history["date"] < pd.Timestamp(cutoff)
    ].sort_values("date")

    records = {}

    for row in prior.itertuples(index=False):
        hg, ag = int(row.home_goals), int(row.away_goals)

        hp = 3 if hg > ag else 1 if hg == ag else 0
        ap = 3 if ag > hg else 1 if ag == hg else 0

        rec_h = records.setdefault(
            row.home_team,
            {"all": [], "home": [], "away": []}
        )

        rec_a = records.setdefault(
            row.away_team,
            {"all": [], "home": [], "away": []}
        )

        rec_h["all"].append((hg, ag, hp))
        rec_h["home"].append((hg, ag, hp))

        rec_a["all"].append((ag, hg, ap))
        rec_a["away"].append((ag, hg, ap))

    hrec = records.get(
        home, {"all": [], "home": [], "away": []}
    )

    arec = records.get(
        away, {"all": [], "home": [], "away": []}
    )

    hf, af = hrec["all"][-5:], arec["all"][-5:]
    hh, aa = hrec["home"][-5:], arec["away"][-5:]

    def avg(items, idx, default):
        return (
            sum(x[idx] for x in items) / len(items)
            if items else default
        )

    return {
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
    }

def bundesliga_teams():
    """Return Bundesliga teams available in historical data."""

    history = load_bundesliga_history()

    return sorted(
        set(history.home_team.dropna())
        | set(history.away_team.dropna())
    )
def predict_bundesliga_match(bundle, home, away, match_date):
    """Predict Bundesliga match using its own trained model."""

    history = load_bundesliga_history()

    home_exists = (
        (history.home_team == home).any()
        or (history.away_team == home).any()
    )

    away_exists = (
        (history.home_team == away).any()
        or (history.away_team == away).any()
    )

    if not home_exists or not away_exists:
        return None

    row = bundesliga_features(
        history, home, away, match_date
    )

    feature_names = bundle.get(
        "features", BUNDESLIGA_FEATURES
    )

    x = pd.DataFrame([row])[feature_names]

    model = bundle["model"]
    probabilities = model.predict_proba(x)[0]

    result = {}

    classes = bundle.get("classes", list(model.classes_))

    for cls, probability in zip(model.classes_, probabilities):
        result[CLASS_NAMES.get(cls, str(cls).lower())] = float(
            probability
        )

    return {
        "home": result.get("home", 0.0),
        "draw": result.get("draw", 0.0),
        "away": result.get("away", 0.0)
    }


def predict_bundesliga_match(bundle, home, away, match_date):
    """Predict Bundesliga match using its own trained model."""

    history = load_bundesliga_history()

    # API names mapped to names used in training history
    team_aliases = {
        "1. FC Köln": "FC Koln",
        "1. FSV Mainz 05": "Mainz",
        "1. FC Union Berlin": "Union Berlin",
        "SC Paderborn 07": "Paderborn",
        "FC Schalke 04": "Schalke 04",
        "Hamburger SV": "Hamburg",
        "FC Bayern München": "Bayern Munich",
        "VfB Stuttgart": "Stuttgart",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "SC Freiburg": "Freiburg",
        "TSG 1899 Hoffenheim": "Hoffenheim",
        "FC Augsburg": "Augsburg",
        "Borussia Dortmund": "Dortmund",
        "Bayer 04 Leverkusen": "Leverkusen",
        "Borussia Mönchengladbach": "M'gladbach",
        "Werder Bremen": "Werder Bremen",
        "RB Leipzig": "RB Leipzig",
    }

    home = team_aliases.get(home, home)
    away = team_aliases.get(away, away)

    home_exists = (
        (history.home_team == home).any()
        or (history.away_team == home).any()
    )

    away_exists = (
        (history.home_team == away).any()
        or (history.away_team == away).any()
    )

    if not home_exists or not away_exists:
        return None

    row = bundesliga_features(
        history, home, away, match_date
    )

    feature_names = bundle.get(
        "features", BUNDESLIGA_FEATURES
    )

    x = pd.DataFrame([row])[feature_names]

    model = bundle["model"]
    probabilities = model.predict_proba(x)[0]

    result = {}

    for cls, probability in zip(model.classes_, probabilities):
        result[CLASS_NAMES.get(cls, str(cls).lower())] = float(
            probability
        )

    return {
        "home": result.get("home", 0.0),
        "draw": result.get("draw", 0.0),
        "away": result.get("away", 0.0)
    }


# =========================================================
# LIGUE 1 ENGINE
# Uses the same feature-generation rules and historical seasons as
# train_ligue1.py. Reads local F1 CSV files; no retraining here.
# =========================================================

LIGUE1_FEATURES = [
    "home_avg_gf5", "home_avg_ga5", "home_points5",
    "away_avg_gf5", "away_avg_ga5", "away_points5",
    "home_home_gf5", "home_home_ga5",
    "away_away_gf5", "away_away_ga5",
]

LIGUE1_DATA_DIR = (
    Path(__file__).resolve().parent / "ligue1_training" / "data" / "raw"
)
LIGUE1_SEASONS = [
    "9394", "9495", "9596", "9697", "9798", "9899", "9900",
    "0001", "0102", "0203", "0304", "0405", "0506", "0607",
    "0708", "0809", "0910", "1011", "1112", "1213", "1314",
    "1415", "1516", "1617", "1718", "1819", "1920", "2021",
    "2122", "2223", "2324", "2425", "2526",
]

# Keep historical club spellings consistent with train_ligue1.py.
LIGUE1_TEAM_ALIASES = {
    "Le Mans UC 72": "Le Mans",
    "MUC 72": "Le Mans",
    "Le Mans UC": "Le Mans",
    "Paris FC": "Paris FC",
}


def _normalize_ligue1_team(value):
    name = str(value).strip()
    return LIGUE1_TEAM_ALIASES.get(name, name)


@lru_cache(maxsize=1)
def load_ligue1_history():
    """Load all locally downloaded Ligue 1 F1 seasons using training's parser."""
    frames = []
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    for season in LIGUE1_SEASONS:
        file_path = LIGUE1_DATA_DIR / f"F1_{season}.csv"
        if not file_path.exists():
            continue

        try:
            try:
                df = pd.read_csv(file_path, encoding="latin-1")
            except (pd.errors.ParserError, UnicodeDecodeError):
                # Match the tolerant fallback used by train_ligue1.py.
                df = pd.read_csv(
                    file_path, encoding="latin-1", engine="python",
                    on_bad_lines="skip"
                )

            if not all(column in df.columns for column in required):
                continue

            df = df[required].copy()
            df["Date"] = pd.to_datetime(
                df["Date"], dayfirst=True, errors="coerce"
            )
            df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
            df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
            df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
            df = df.dropna(
                subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            )
            df = df[df["FTR"].isin(["H", "D", "A"])].copy()
            df["FTHG"] = df["FTHG"].astype(int)
            df["FTAG"] = df["FTAG"].astype(int)
            df["HomeTeam"] = df["HomeTeam"].map(_normalize_ligue1_team)
            df["AwayTeam"] = df["AwayTeam"].map(_normalize_ligue1_team)
            frames.append(df)
        except Exception as exc:
            print(f"Ligue 1 history: skipped {file_path.name}: {exc}")
            continue

    if not frames:
        raise RuntimeError(
            f"No usable Ligue 1 F1 CSV files found in {LIGUE1_DATA_DIR}"
        )

    history = pd.concat(frames, ignore_index=True)
    return history.sort_values("Date").reset_index(drop=True)


def _ligue1_blank_record():
    return {
        "gf": [], "ga": [], "pts": [],
        "home_gf": [], "home_ga": [],
        "away_gf": [], "away_ga": [],
    }


def _ligue1_avg_last(values, default):
    values = values[-5:]
    return float(sum(values) / len(values)) if values else float(default)


def ligue1_features(history, home, away, cutoff):
    """Create the exact ten rolling features used by train_ligue1.py."""
    cutoff = pd.Timestamp(cutoff)
    teams = {}

    # Iterate through past matches only, preserving the training pipeline's
    # chronological update order and excluding the target match itself.
    prior = history[history["Date"] < cutoff].sort_values("Date")

    for row in prior.itertuples(index=False):
        home_name, away_name = str(row.HomeTeam), str(row.AwayTeam)
        h = teams.setdefault(home_name, _ligue1_blank_record())
        a = teams.setdefault(away_name, _ligue1_blank_record())

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

    h = teams.get(home, _ligue1_blank_record())
    a = teams.get(away, _ligue1_blank_record())

    return {
        "home_avg_gf5": _ligue1_avg_last(h["gf"], 1.35),
        "home_avg_ga5": _ligue1_avg_last(h["ga"], 1.35),
        "home_points5": _ligue1_avg_last(h["pts"], 1.0),
        "away_avg_gf5": _ligue1_avg_last(a["gf"], 1.10),
        "away_avg_ga5": _ligue1_avg_last(a["ga"], 1.45),
        "away_points5": _ligue1_avg_last(a["pts"], 1.0),
        "home_home_gf5": _ligue1_avg_last(h["home_gf"], 1.35),
        "home_home_ga5": _ligue1_avg_last(h["home_ga"], 1.35),
        "away_away_gf5": _ligue1_avg_last(a["away_gf"], 1.10),
        "away_away_ga5": _ligue1_avg_last(a["away_ga"], 1.45),
    }


def ligue1_teams():
    """Return team names exactly as stored in the Ligue 1 training data."""
    history = load_ligue1_history()
    return sorted(
        set(history["HomeTeam"].dropna())
        | set(history["AwayTeam"].dropna())
    )


def predict_ligue1_match(bundle, home, away, match_date):
    """Predict a Ligue 1 fixture using the separately trained Ligue 1 model."""
    history = load_ligue1_history()
    home = _normalize_ligue1_team(home)
    away = _normalize_ligue1_team(away)
    known = set(history["HomeTeam"].dropna()) | set(history["AwayTeam"].dropna())

    if home not in known or away not in known:
        return None

    row = ligue1_features(history, home, away, match_date)
    feature_names = bundle.get("features", LIGUE1_FEATURES)
    x = pd.DataFrame([row])[feature_names]

    model = bundle["model"]
    probabilities = model.predict_proba(x)[0]
    result = {}

    for cls, probability in zip(model.classes_, probabilities):
        result[CLASS_NAMES.get(cls, str(cls).lower())] = float(probability)

    return {
        "home": result.get("home", 0.0),
        "draw": result.get("draw", 0.0),
        "away": result.get("away", 0.0),
    }
