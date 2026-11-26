
import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "raw" /
    "Home_Advantage_Enriched.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "caf_matches.csv"

print("=" * 55)
print("CAF CHAMPIONS LEAGUE DATA CLEANING")
print("=" * 55)

# 1. Check input file
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

# 2. Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nOriginal matches:", len(df))

# 3. Select CAF Champions League only
df = df[
    df["Competition"].astype(str).str.strip()
    == "CAF Champions League"
].copy()

print("CAF matches found:", len(df))

# 4. Select useful columns
columns = [
    "Date",
    "Season",
    "Competition",
    "Stage",
    "Home Team",
    "Away Team",
    "Home Goals",
    "Away Goals",
    "Neutral",
    "Match Result",
    "Travel Distance (km)",
    "Altitude Difference (m)",
    "Climate Zone"
]

df = df[columns].copy()

# 5. Clean dates
df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

# 6. Convert goals to numeric
df["Home Goals"] = pd.to_numeric(
    df["Home Goals"], errors="coerce"
)

df["Away Goals"] = pd.to_numeric(
    df["Away Goals"], errors="coerce"
)

# 7. Remove invalid records
before = len(df)

df = df.dropna(
    subset=[
        "Date",
        "Home Team",
        "Away Team",
        "Home Goals",
        "Away Goals"
    ]
).copy()

print("Invalid records removed:", before - len(df))

# 8. Standardize team names
df["Home Team"] = df["Home Team"].astype(str).str.strip()
df["Away Team"] = df["Away Team"].astype(str).str.strip()

# 9. Convert goals to integers
df["Home Goals"] = df["Home Goals"].astype(int)
df["Away Goals"] = df["Away Goals"].astype(int)

# 10. Generate prediction target
def get_result(row):
    if row["Home Goals"] > row["Away Goals"]:
        return "H"
    elif row["Home Goals"] == row["Away Goals"]:
        return "D"
    else:
        return "A"

df["Result"] = df.apply(get_result, axis=1)

# 11. Check original result consistency
result_mapping = {
    "Home Win": "H",
    "Draw": "D",
    "Away Win": "A"
}

df["Original Result"] = (
    df["Match Result"].astype(str).str.strip()
    .map(result_mapping)
)

mismatches = (
    df["Original Result"].notna()
    & (df["Original Result"] != df["Result"])
)

print("Result inconsistencies:", int(mismatches.sum()))

# 12. Remove exact duplicate records
before = len(df)

df = df.drop_duplicates(
    subset=[
        "Date", "Home Team", "Away Team",
        "Home Goals", "Away Goals"
    ]
)

print("Duplicate records removed:", before - len(df))

# 13. Rename columns
df = df.rename(columns={
    "Date": "date",
    "Season": "season",
    "Competition": "competition",
    "Stage": "stage",
    "Home Team": "home_team",
    "Away Team": "away_team",
    "Home Goals": "home_goals",
    "Away Goals": "away_goals",
    "Neutral": "neutral",
    "Travel Distance (km)": "travel_distance_km",
    "Altitude Difference (m)": "altitude_difference_m",
    "Climate Zone": "climate_zone"
})

# 14. Sort matches by date
df = df.sort_values("date").reset_index(drop=True)

# 15. Save processed dataset
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

# 16. Display summary
print("\n" + "=" * 55)
print("CLEANING COMPLETED")
print("=" * 55)

print("Final matches:", len(df))
print("\nResult distribution:")
print(df["Result"].value_counts())

print("\nFirst 5 matches:")
print(
    df[
        ["date", "home_team", "away_team",
         "home_goals", "away_goals", "Result"]
    ].head()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nSUCCESS!")
