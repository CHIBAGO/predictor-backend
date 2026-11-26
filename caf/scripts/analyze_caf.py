
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

FILE = BASE_DIR / "data" / "processed" / "caf_matches.csv"

print("=" * 60)
print("CAF CHAMPIONS LEAGUE - DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"])

# 1. General information
print("\n1. GENERAL INFORMATION")

print("Total matches:", len(df))
print("First match:", df["date"].min().date())
print("Last match:", df["date"].max().date())

print("Unique home teams:", df["home_team"].nunique())
print("Unique away teams:", df["away_team"].nunique())

all_teams = set(df["home_team"]) | set(df["away_team"])

print("Total unique teams:", len(all_teams))

# 2. Season analysis
print("\n2. MATCHES PER SEASON")

print(df["season"].value_counts().sort_index())

# 3. Stage analysis
print("\n3. MATCHES PER STAGE")

print(df["stage"].value_counts())

# 4. Result distribution
print("\n4. RESULT DISTRIBUTION")

results = df["Result"].value_counts()

for result, count in results.items():
    percentage = count / len(df) * 100

    name = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    }.get(result, result)

    print(f"{name}: {count} ({percentage:.2f}%)")

# 5. Most active teams
print("\n5. TOP 15 MOST ACTIVE TEAMS")

team_matches = pd.concat([
    df["home_team"],
    df["away_team"]
]).value_counts()

print(team_matches.head(15))

# 6. Goal analysis
print("\n6. GOAL ANALYSIS")

print("Total home goals:", df["home_goals"].sum())
print("Total away goals:", df["away_goals"].sum())

print(
    "Average home goals:",
    round(df["home_goals"].mean(), 2)
)

print(
    "Average away goals:",
    round(df["away_goals"].mean(), 2)
)

print(
    "Average total goals per match:",
    round(
        (df["home_goals"] + df["away_goals"]).mean(),
        2
    )
)

# 7. Neutral venue analysis
print("\n7. NEUTRAL VENUE ANALYSIS")

print(df["neutral"].value_counts(dropna=False))

# 8. Result by competition stage
print("\n8. RESULTS BY STAGE")

print(
    pd.crosstab(
        df["stage"],
        df["Result"],
        margins=True
    )
)

# 9. Chronological split
print("\n9. TEMPORAL DATA SPLIT")

df = df.sort_values("date").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

print("Training matches:", len(train))
print("Testing matches:", len(test))

print("Training period:",
      train["date"].min().date(),
      "to",
      train["date"].max().date())

print("Testing period:",
      test["date"].min().date(),
      "to",
      test["date"].max().date())

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)
