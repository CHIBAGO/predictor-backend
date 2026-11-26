
import pandas as pd
from pathlib import Path
from collections import defaultdict
import math

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "caf_matches.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"

INITIAL_ELO = 1500
HOME_ADVANTAGE = 60
ELO_K = 20

print("=" * 65)
print("CAF CHAMPIONS LEAGUE - FEATURE ENGINEERING V2")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Team match history
history = defaultdict(list)

# Elo ratings
elo = defaultdict(lambda: INITIAL_ELO)

# Head-to-head history
h2h_history = defaultdict(list)


def average(values, default=0):
    return sum(values) / len(values) if values else default


def get_team_features(team, venue):
    records = history[team]
    last5 = records[-5:]

    venue_records = [
        record for record in records if record["venue"] == venue
    ][-5:]

    return {
        "avg_gf5": average([r["gf"] for r in last5], 1.2),
        "avg_ga5": average([r["ga"] for r in last5], 1.2),
        "points5": average([r["points"] for r in last5], 1.0),
        "venue_gf5": average([r["gf"] for r in venue_records], 1.2),
        "venue_ga5": average([r["ga"] for r in venue_records], 1.2),
        "matches_played": len(records),

        # New V2 features
        "win_rate5": average(
            [r["win"] for r in last5], 0.33
        ),
        "goal_difference5": average(
            [r["gf"] - r["ga"] for r in last5], 0
        )
    }


def get_h2h_features(home, away):
    pair = tuple(sorted([home, away]))
    records = h2h_history[pair][-5:]

    home_points = []
    home_goal_differences = []

    for record in records:
        if record["team_a"] == home:
            gf = record["goals_a"]
            ga = record["goals_b"]
            points = record["points_a"]
        else:
            gf = record["goals_b"]
            ga = record["goals_a"]
            points = record["points_b"]

        home_points.append(points)
        home_goal_differences.append(gf - ga)

    return {
        "h2h_matches": len(records),
        "h2h_home_points5": average(home_points, 1.0),
        "h2h_home_goal_difference5": average(
            home_goal_differences, 0
        )
    }


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


output = []

# Process matches by date to prevent same-day information leakage
for match_date, group in df.groupby("date", sort=True):

    current_matches = group.to_dict("records")

    # First calculate features for all matches on this date
    for row in current_matches:

        home = row["home_team"]
        away = row["away_team"]

        home_features = get_team_features(home, "home")
        away_features = get_team_features(away, "away")
        h2h = get_h2h_features(home, away)

        home_elo = elo[home]
        away_elo = elo[away]

        # Home advantage is added only when calculating expectation
        expected_home = expected_score(
            home_elo + HOME_ADVANTAGE,
            away_elo
        )

        expected_away = 1 - expected_home

        output.append({
            "date": row["date"],
            "season": row["season"],
            "stage": row["stage"],
            "home_team": home,
            "away_team": away,

            # Existing V1 features
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

            # New V2 form features
            "home_win_rate5": home_features["win_rate5"],
            "away_win_rate5": away_features["win_rate5"],
            "home_goal_difference5": home_features["goal_difference5"],
            "away_goal_difference5": away_features["goal_difference5"],

            # New V2 Elo features
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo + HOME_ADVANTAGE - away_elo,
            "expected_home_score": expected_home,
            "expected_away_score": expected_away,

            # New V2 head-to-head features
            "h2h_matches": h2h["h2h_matches"],
            "h2h_home_points5": h2h["h2h_home_points5"],
            "h2h_home_goal_difference5":
                h2h["h2h_home_goal_difference5"],

            # Target values
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "result": row["Result"]
        })

    # Update histories only after all matches on this date
    for row in current_matches:

        home = row["home_team"]
        away = row["away_team"]

        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        if hg > ag:
            home_points, away_points = 3, 0
            home_win, away_win = 1, 0
            actual_home = 1
        elif hg == ag:
            home_points, away_points = 1, 1
            home_win, away_win = 0, 0
            actual_home = 0.5
        else:
            home_points, away_points = 0, 3
            home_win, away_win = 0, 1
            actual_home = 0

        # Update team form history
        history[home].append({
            "gf": hg,
            "ga": ag,
            "points": home_points,
            "win": home_win,
            "venue": "home"
        })

        history[away].append({
            "gf": ag,
            "ga": hg,
            "points": away_points,
            "win": away_win,
            "venue": "away"
        })

        # Update Elo ratings
        old_home_elo = elo[home]
        old_away_elo = elo[away]

        expected_home = expected_score(
            old_home_elo + HOME_ADVANTAGE,
            old_away_elo
        )

        expected_away = 1 - expected_home

        new_home_elo = old_home_elo + ELO_K * (
            actual_home - expected_home
        )

        new_away_elo = old_away_elo + ELO_K * (
            (1 - actual_home) - expected_away
        )

        elo[home] = new_home_elo
        elo[away] = new_away_elo

        # Store head-to-head history
        pair = tuple(sorted([home, away]))

        if pair[0] == home:
            goals_a, goals_b = hg, ag
            points_a, points_b = home_points, away_points
        else:
            goals_a, goals_b = ag, hg
            points_a, points_b = away_points, home_points

        h2h_history[pair].append({
            "team_a": pair[0],
            "team_b": pair[1],
            "goals_a": goals_a,
            "goals_b": goals_b,
            "points_a": points_a,
            "points_b": points_b
        })


# Save V2 dataset
features = pd.DataFrame(output)
features.to_csv(OUTPUT_FILE, index=False)

print("\nFEATURE ENGINEERING V2 COMPLETED")
print(f"Total matches: {len(features)}")
print(f"Total columns: {len(features.columns)}")

print("\nNew V2 features:")
new_features = [
    "home_win_rate5",
    "away_win_rate5",
    "home_goal_difference5",
    "away_goal_difference5",
    "home_elo",
    "away_elo",
    "elo_difference",
    "expected_home_score",
    "expected_away_score",
    "h2h_matches",
    "h2h_home_points5",
    "h2h_home_goal_difference5"
]

for feature in new_features:
    print(f"- {feature}")

print("\nFirst five matches:")
print(features.head().to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)
print("\nSUCCESS!")
