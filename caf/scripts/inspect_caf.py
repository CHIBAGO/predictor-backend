
import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]

FILE = BASE_DIR / "data" / "raw" / "Home_Advantage_Enriched.csv"

print("=" * 60)
print("CAF DATASET INSPECTION")
print("=" * 60)

if not FILE.exists():
    print("ERROR: Dataset not found:", FILE)
    raise SystemExit(1)

with open(FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    rows = list(reader)
    columns = reader.fieldnames or []

print("\nFILE:", FILE)
print("TOTAL MATCHES:", len(rows))

print("\nAVAILABLE COLUMNS:")
for column in columns:
    print("-", column)

print("\nFIRST 5 MATCHES:")
for index, row in enumerate(rows[:5], 1):
    print(f"\nMATCH {index}")
    for key, value in row.items():
        print(f"  {key}: {value}")

print("\nPOSSIBLE COMPETITION COLUMNS:")

for column in columns:
    if any(word in column.lower() for word in
           ["competition", "league", "tournament", "cup"]):

        counts = Counter(
            row[column] for row in rows if row.get(column)
        )

        print(f"\n{column}:")
        for name, count in counts.most_common(30):
            print(f"  {name}: {count}")

print("\nINSPECTION COMPLETED")
