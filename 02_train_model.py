"""
PREDICTOR - Step 2: Teach the AI
================================
What this file does:
  1. Reads the real matches saved by 01_get_data.py
  2. For every match, builds the numbers the AI will look at (ratings, form, rest days)
  3. Trains two models and tests them on the MOST RECENT matches (which they never saw)
  4. Compares them with a simple baseline, and saves the best one

How to run (core.py must be in the same folder):
    py -m pip install pandas scikit-learn joblib
    py 02_train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import FEATURES, build_training_table, load_matches

MODEL_FILE = "predictor_model.joblib"
LABELS = ["A", "D", "H"]            # away win, draw, home win (alphabetical order)

# ---------------------------------------------------------------------
# 1 + 2. Load the matches and build the numbers
# ---------------------------------------------------------------------
matches = load_matches()
print(f"Loaded {len(matches)} matches, {matches['season'].iloc[0]} to {matches['season'].iloc[-1]}")

table, state = build_training_table(matches)

# The first season is only used to "warm up" the ratings, so we do not train on it.
# We also skip matches where a team has fewer than 5 matches of history.
first_season = table["season"].iloc[0]
table = table[(table["season"] != first_season) & (table["h_played"] >= 5) & (table["a_played"] >= 5)]
print(f"Matches the AI can use: {len(table)}")

# ---------------------------------------------------------------------
# 3. Split by TIME: old matches to learn from, the latest two seasons to test.
#    (We never mix them randomly. In real life we predict the future from the past.)
# ---------------------------------------------------------------------
seasons = list(dict.fromkeys(table["season"]))           # seasons in order
test_seasons = seasons[-2:]
train = table[~table["season"].isin(test_seasons)]
test = table[table["season"].isin(test_seasons)]
print(f"Learning from {len(train)} matches ({seasons[0]} to {seasons[-3]})")
print(f"Testing on {len(test)} matches ({test_seasons[0]} and {test_seasons[1]})")

X_train, y_train = train[FEATURES], train["result"]
X_test, y_test = test[FEATURES], test["result"]

# ---------------------------------------------------------------------
# 4. The baseline: what we must beat
# ---------------------------------------------------------------------
shares = y_train.value_counts(normalize=True)
baseline_probs = np.tile([shares.get(c, 0.0) for c in LABELS], (len(test), 1))
baseline_accuracy = (y_test == shares.idxmax()).mean()
baseline_loss = log_loss(y_test, baseline_probs, labels=LABELS)

# ---------------------------------------------------------------------
# 5. Train two models
# ---------------------------------------------------------------------
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=15, random_state=42),
}

results = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "log_loss": log_loss(y_test, probs, labels=LABELS),
        "probs": probs,
    }

print("\n" + "=" * 62)
print(f"{'Model':<22}{'Accuracy':>12}{'Log loss':>14}")
print("-" * 62)
print(f"{'Baseline (guess '+shares.idxmax()+')':<22}{baseline_accuracy * 100:>11.1f}%{baseline_loss:>14.3f}")
for name, r in results.items():
    print(f"{name:<22}{r['accuracy'] * 100:>11.1f}%{r['log_loss']:>14.3f}")
print("=" * 62)
print("Accuracy: how often the top guess was right (higher is better).")
print("Log loss: how good the probabilities are (LOWER is better).")

best_name = min(results, key=lambda n: results[n]["log_loss"])
best = results[best_name]
print(f"\nBest model: {best_name}")
if best["log_loss"] < baseline_loss:
    print("It beats the baseline. The AI has learned something real.")
else:
    print("It does NOT beat the baseline. We need better numbers or more data.")

# ---------------------------------------------------------------------
# 6. Can we trust the probabilities?  (when the AI says 60%, is it right ~60% of the time?)
# ---------------------------------------------------------------------
top_prob = best["probs"].max(axis=1)
top_guess = np.array(LABELS)[best["probs"].argmax(axis=1)]
correct = (top_guess == y_test.to_numpy())
print("\nWhen the AI is this confident ... how often is it right?")
for low, high in [(0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
    mask = (top_prob >= low) & (top_prob < high)
    if mask.sum() >= 5:
        print(f"  {low * 100:>3.0f}% to {min(high, 1) * 100:>3.0f}% : right {correct[mask].mean() * 100:>5.1f}% of the time  ({mask.sum()} matches)")

# ---------------------------------------------------------------------
# 7. Save the best model. We train it again on ALL the data first,
#    so it also knows the latest matches.
# ---------------------------------------------------------------------
final_model = candidates[best_name]
final_model.fit(table[FEATURES], table["result"])

bundle = {
    "model": final_model,
    "features": FEATURES,
    "state": state,                                   # current ratings and form of every team
    "info": {
        "model_name": best_name,
        "trained_on_matches": int(len(table)),
        "last_match_date": str(matches["date"].iloc[-1].date()),
        "test_accuracy": float(best["accuracy"]),
        "baseline_accuracy": float(baseline_accuracy),
        "test_log_loss": float(best["log_loss"]),
        "baseline_log_loss": float(baseline_loss),
    },
}
joblib.dump(bundle, MODEL_FILE)
print(f"\nSaved the model in {MODEL_FILE}")

# ---------------------------------------------------------------------
# 8. A quick sanity check: do the ratings make football sense?
# ---------------------------------------------------------------------
print("\nTop 5 teams by rating right now:")
for team, rating in state.current_ratings()[:5]:
    print(f"  {team:<22}{rating:>7.0f}")
print("Bottom 3:")
for team, rating in state.current_ratings()[-3:]:
    print(f"  {team:<22}{rating:>7.0f}")
