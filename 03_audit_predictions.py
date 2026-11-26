"""Predictor diagnostic: verify probability/accuracy/log-loss consistency."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import FEATURES, build_training_table, load_matches

LABELS = ["A", "D", "H"]

matches = load_matches()
table, state = build_training_table(matches)
first_season = table["season"].iloc[0]
table = table[(table["season"] != first_season) & (table["h_played"] >= 5) & (table["a_played"] >= 5)]
seasons = list(dict.fromkeys(table["season"]))
test_seasons = seasons[-2:]
train = table[~table["season"].isin(test_seasons)]
test = table[table["season"].isin(test_seasons)]
X_train, y_train = train[FEATURES], train["result"]
X_test, y_test = test[FEATURES], test["result"]

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=15, random_state=42),
}
print(f"Test matches: {len(test)} | Test seasons: {test_seasons}")
print(f"Labels in test: {sorted(y_test.unique())}")
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    classes = list(model.classes_)
    preds_from_probs = np.array(classes)[probs.argmax(axis=1)]
    preds_model = model.predict(X_test)
    acc_from_probs = accuracy_score(y_test, preds_from_probs)
    acc_model = accuracy_score(y_test, preds_model)
    loss = log_loss(y_test, probs, labels=classes)
    true_col = np.array([classes.index(label) for label in y_test])
    true_probs = probs[np.arange(len(y_test)), true_col]
    min_possible_loss = -np.log(np.maximum(true_probs, 1e-15)).mean()
    print("\n" + "="*55)
    print(name)
    print("classes_ order:", classes)
    print("Accuracy from probabilities:", round(acc_from_probs, 6))
    print("Accuracy from model.predict:", round(acc_model, 6))
    print("Log loss:", round(loss, 6))
    print("Mean -log(true-class probability):", round(min_possible_loss, 6))
    print("Probability min/max:", float(probs.min()), float(probs.max()))
    print("Row probability sums min/max:", float(probs.sum(axis=1).min()), float(probs.sum(axis=1).max()))
    print("Prediction disagreement count:", int(np.sum(preds_from_probs != preds_model)))
    print("Confusion matrix (rows actual, columns predicted; order", classes, "):")
    print(confusion_matrix(y_test, preds_from_probs, labels=classes))
    print("Correctness by confidence:")
    confidence = probs.max(axis=1)
    correct = preds_from_probs == y_test.to_numpy()
    for lo, hi in [(0, .45), (.45, .55), (.55, .65), (.65, 1.000001)]:
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum():
            print(f"  {lo:.0%}-{min(hi,1):.0%}: {correct[mask].mean():.1%} correct / {mask.sum()} matches")
