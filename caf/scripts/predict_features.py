
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "caf_matches.csv"

INITIAL_ELO = 1500
HOME_ADVANTAGE = 60
ELO_K = 20


def average(values, default=0):
    return sum(values) / len(values) if values else default


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def get_team_features(history, team, venue):
    records = history[team]
    last5 = records[-5:]

    venue_records = [
        r for r in records if r["venue"] == venue
    ][-5:]

    return {
        "avg_gf5": average([r["gf"] for r in last5], 1.2),
        "avg_ga5": average([r["ga"] for r in last5], 1.2),
        "points5": average([r["points"] for r in last5], 1.0),
        "venue_gf5": average([r["gf"] for r in venue_records], 1.2),
        "venue_ga5": average([r["ga"] for r in venue_records], 1.2),
        "matches_played": len(records),
        "win_rate5": average([r["win"] for r in last5], 0.33),
        "goal_difference5": average(
            [r["gf"] - r["ga"] for r in last5], 0
        )
    }


def get_h2h_features(h2h_history, home, away):
    pair = tuple(sorted([home, away]))
    records = h2h_history[pair][-5:]

    points = []
    goal_differences = []

    for record in records:
        if record["team_a"] == home:
            gf = record["goals_a"]
            ga = record["goals_b"]
            pts = record["points_a"]
        else:
            gf = record["goals_b"]
            ga = record["goals_a"]
            pts = record["points_b"]

        points.append(pts)
        goal_differences.append(gf - ga)

    return {
        "h2h_matches": len(records),
        "h2h_home_points5": average(points, 1.0),
        "h2h_home_goal_difference5": average(
            goal_differences, 0
        )
    }


def build_prediction_features(
    home, away, match_date, stage="Group Stage"
):
    if home == away:
        raise ValueError("Home and away teams must be different.")

    match_date = pd.to_datetime(match_date)

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Use only matches played before the prediction date
    df = df[df["date"] < match_date]

    if df.empty:
        raise ValueError(
            "No historical matches available before this date."
        )

    history = defaultdict(list)
    elo = defaultdict(lambda: INITIAL_ELO)
    h2h_history = defaultdict(list)

    # Rebuild historical team information
    for row in df.to_dict("records"):
        home_team = row["home_team"]
        away_team = row["away_team"]

        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        if hg > ag:
            hp, ap = 3, 0
            hw, aw = 1, 0
            actual_home = 1
        elif hg == ag:
            hp, ap = 1, 1
            hw, aw = 0, 0
            actual_home = 0.5
        else:
            hp, ap = 0, 3
            hw, aw = 0, 1
            actual_home = 0

        history[home_team].append({
            "gf": hg, "ga": ag,
            "points": hp, "win": hw, "venue": "home"
        })

        history[away_team].append({
            "gf": ag, "ga": hg,
            "points": ap, "win": aw, "venue": "away"
        })

        old_home = elo[home_team]
        old_away = elo[away_team]

        exp_home = expected_score(
            old_home + HOME_ADVANTAGE, old_away
        )
        exp_away = 1 - exp_home

        elo[home_team] = old_home + ELO_K * (
            actual_home - exp_home
        )
        elo[away_team] = old_away + ELO_K * (
            (1 - actual_home) - exp_away
        )

        pair = tuple(sorted([home_team, away_team]))

        if pair[0] == home_team:
            goals_a, goals_b = hg, ag
            points_a, points_b = hp, ap
        else:
            goals_a, goals_b = ag, hg
            points_a, points_b = ap, hp

        h2h_history[pair].append({
            "team_a": pair[0],
            "team_b": pair[1],
            "goals_a": goals_a,
            "goals_b": goals_b,
            "points_a": points_a,
            "points_b": points_b
        })

    home_features = get_team_features(history, home, "home")
    away_features = get_team_features(history, away, "away")
    h2h = get_h2h_features(h2h_history, home, away)

    home_elo = elo[home]
    away_elo = elo[away]

    exp_home = expected_score(
        home_elo + HOME_ADVANTAGE, away_elo
    )
    exp_away = 1 - exp_home

    features = {
        "stage": stage,

        "home_avg_gf5": home_features["avg_gf5"],
        "home_avg_ga5": home_features["avg_ga5"],
        "home_points5": home_features["points5"],
        "home_home_gf5": home_features["venue_gf5"],
        "home_home_ga5": home_features["venue_ga5"],
        "home_matches_played": home_features["matches_played"],

        "away_avg_gf5": away_features["avg_gf5"],
        "away_avg_ga5": away_features["avg_ga5"],
        "away_points5": away_features["points5"],
        "away_away_gf5": away_features["venue_gf5"],
        "away_away_ga5": away_features["venue_ga5"],
        "away_matches_played": away_features["matches_played"],

        "home_win_rate5": home_features["win_rate5"],
        "away_win_rate5": away_features["win_rate5"],
        "home_goal_difference5": home_features["goal_difference5"],
        "away_goal_difference5": away_features["goal_difference5"],

        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo + HOME_ADVANTAGE - away_elo,
        "expected_home_score": exp_home,
        "expected_away_score": exp_away,

        "h2h_matches": h2h["h2h_matches"],
        "h2h_home_points5": h2h["h2h_home_points5"],
        "h2h_home_goal_difference5":
            h2h["h2h_home_goal_difference5"]
    }

    return features


if __name__ == "__main__":
    home = input("Enter home team: ").strip()
    away = input("Enter away team: ").strip()
    match_date = input("Enter match date (YYYY-MM-DD): ").strip()
    stage = input("Enter stage (default Group Stage): ").strip()

    if not stage:
        stage = "Group Stage"

    try:
        features = build_prediction_features(
            home, away, match_date, stage
        )

        print("\nCAF PREDICTION FEATURES")
        print("=" * 50)

        for name, value in features.items():
            print(f"{name}: {value}")

        print("\nTotal features:", len(features))
        print("\nSUCCESS!")

    except Exception as error:
        print("\nERROR:", error)
