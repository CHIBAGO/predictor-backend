
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ucl" / "data" / "raw"

print("========== CURRENT SEASON CSV FILES ==========")

current = RAW / "open_matchday_2627"

for filename in ["results_2627.csv", "fixtures_2627.csv", "goals_2627.csv"]:
    path = current / filename
    print(f"\n--- {filename} ---")

    if path.exists():
        df = pd.read_csv(path)
        print("Columns:", list(df.columns))
        print("Rows:", len(df))
        print(df.head(3).to_string(index=False))
    else:
        print("File not found:", path)

print("\n========== HISTORICAL FILES ==========")

historical = RAW / "openfootball_historical"

for path in sorted(historical.rglob("*.txt"))[:5]:
    print(f"\n--- {path.relative_to(historical)} ---")
    with open(path, encoding="utf-8", errors="replace") as file:
        for _ in range(12):
            line = file.readline()
            if not line:
                break
            print(line.rstrip())
