
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "caf_features_v2.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("CAF FEATURE AUDIT")
print("=" * 70)

print("\n1. DATASET INFORMATION")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. FEATURE LIST")
for column in df.columns:
    print("-", column)

print("\n3. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

print("\n4. DUPLICATE ROWS")
print("Duplicates:", df.duplicated().sum())

print("\n5. DATA TYPES")
print(df.dtypes)

print("\n6. RESULT DISTRIBUTION")
print(df["result"].value_counts())

print("\n7. NUMERIC FEATURE STATISTICS")
print(df.select_dtypes(include="number").describe().round(3))

print("\n8. TEAM INFORMATION")
print("Unique home teams:", df["home_team"].nunique())
print("Unique away teams:", df["away_team"].nunique())

print("\n9. FEATURE CORRELATION WITH RESULT")

numeric = df.select_dtypes(include="number").copy()
numeric["result_encoded"] = df["result"].map({
    "H": 0,
    "D": 1,
    "A": 2
})

correlation = numeric.corr()["result_encoded"].drop(
    "result_encoded"
).sort_values(key=abs, ascending=False)

print(correlation.round(3))

print("\n10. DATE RANGE")
print("From:", df["date"].min())
print("To:", df["date"].max())

output = BASE_DIR / "training" / "caf_feature_audit.csv"

audit = pd.DataFrame({
    "feature": df.columns,
    "missing_values": [df[col].isnull().sum() for col in df.columns],
    "unique_values": [df[col].nunique() for col in df.columns],
    "data_type": [str(df[col].dtype) for col in df.columns]
})

audit.to_csv(output, index=False)

print("\nAudit report saved to:", output)
print("\nFEATURE AUDIT COMPLETED!")
