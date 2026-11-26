
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

periods = [
    ("All Matches", None, None),
    ("Before 2020", None, "2020-01-01"),
    ("2020-2021", "2020-01-01", "2022-01-01"),
    ("2022-2023", "2022-01-01", "2024-01-01"),
    ("2024-2025", "2024-01-01", "2026-01-01")
]

print("=" * 65)
print("CAF DATASET - RESULT DISTRIBUTION")
print("=" * 65)

results = []

for name, start, end in periods:

    subset = df.copy()

    if start:
        subset = subset[subset["date"] >= start]

    if end:
        subset = subset[subset["date"] < end]

    counts = subset["result"].value_counts()
    total = len(subset)

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)
    print("Total matches:", total)

    for result, label in [
        ("H", "Home Win"),
        ("D", "Draw"),
        ("A", "Away Win")
    ]:
        count = counts.get(result, 0)
        percentage = (count / total * 100) if total else 0

        print(
            f"{label}: {count} matches ({percentage:.2f}%)"
        )

        results.append({
            "period": name,
            "result": label,
            "matches": count,
            "percentage": round(percentage, 2)
        })

output = BASE_DIR / "training" / "caf_class_distribution.csv"

pd.DataFrame(results).to_csv(output, index=False)

print("\nAnalysis saved to:", output)
print("\nSUCCESS!")
