
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from train import load_dataset, MODEL_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    X_test, y_test = load_dataset("KDDTest+.txt")

    model_path = MODEL_DIR / "decision_tree.joblib"
    model = joblib.load(model_path)

    # Get predicted probabilities for the Attack class (label 1).
    attack_probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        predictions = (attack_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Threshold": threshold,
            "Accuracy (%)": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "Attack Precision (%)": round(
                precision_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "Attack Recall (%)": round(
                recall_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "Attack F1 (%)": round(
                f1_score(y_test, predictions, zero_division=0) * 100, 2
            ),
            "False Alarms": int(fp),
            "Missed Attacks": int(fn)
        })

    results_df = pd.DataFrame(results)

    print("\nDECISION TREE THRESHOLD EXPERIMENT")
    print("=" * 100)
    print(results_df.to_string(index=False))

    results_df.to_csv(
        PROJECT_DIR / "threshold_results.csv", index=False
    )
    print("\nSaved results to threshold_results.csv")


if __name__ == "__main__":
    main()
