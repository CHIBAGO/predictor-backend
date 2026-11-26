
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "caf_matches.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "caf_features.csv"

print("=" * 60)
print("CAF CHAMPIONS LEAGUE - FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# Each team's history:
# (goals_for, goals_against, points, venue)
history = defaultdict(list)

def average(records, index, default):
    if not records:
        return default
    return sum(r[index] for r in records) / len(records)


def get_features(team, venue):
    records = history[team]

    last5 = records[-5:]

    venue_records = [
        r for r in records if r[3] == venue
    ][-5:]

    return {
        "avg_gf5": average(last5, 0, 1.2),
        "avg_ga5": average(last5, 1, 1.2),
        "points5": average(last5, 2, 1.0),
        "venue_gf5": average(venue_records, 0, 1.2),
        "venue_ga5": average(venue_records, 1, 1.2),
        "matches_played": len(records)
    }


output = []

# Process each date separately.
# Matches on the same date cannot use each other's results.
for match_date, group in df.groupby("date", sort=True):

    current_matches = []

    for row in group.to_dict("records"):

        home = row["home_team"]
        away = row["away_team"]

        hf = get_features(home, "home")
        af = get_features(away, "away")

        record = {
            "date": row["date"],
            "season": row["season"],
            "stage": row["stage"],

            "home_team": home,
            "away_team": away,

            "home_avg_gf5": hf["avg_gf5"],
            "home_avg_ga5": hf["avg_ga5"],
            "home_points5": hf["points5"],
            "home_home_gf5": hf["venue_gf5"],
            "home_home_ga5": hf["venue_ga5"],
            "home_matches_played": hf["matches_played"],

            "away_avg_gf5": af["avg_gf5"],
            "away_avg_ga5": af["avg_ga5"],
            "away_points5": af["points5"],
            "away_away_gf5": af["venue_gf5"],
            "away_away_ga5": af["venue_ga5"],
            "away_matches_played": af["matches_played"],

            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "result": row["Result"]
        }

        output.append(record)

        current_matches.append(row)

    # Update histories only after processing the entire date.
    for row in current_matches:

        home = row["home_team"]
        away = row["away_team"]

        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        history[home].append((hg, ag, hp, "home"))
        history[away].append((ag, hg, ap, "away"))


# Create final dataset
features = pd.DataFrame(output)

# Save
features.to_csv(OUTPUT_FILE, index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("Total matches:", len(features))
print("Total features:", len(features.columns))

print("\nFeature columns:")
for column in features.columns:
    print("-", column)

print("\nFirst 5 engineered matches:")
print(features.head().to_string(index=False))

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nSUCCESS!")
