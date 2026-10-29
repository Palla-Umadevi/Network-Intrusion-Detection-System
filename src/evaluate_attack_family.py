
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from train_attack_family import (
    PROJECT_DIR,
    DATASET_DIR,
    MODEL_DIR,
    FEATURE_NAMES,
    COLUMNS,
    get_family
)


def main():
    test_path = DATASET_DIR / "KDDTest+.txt"
    model_path = MODEL_DIR / "attack_family.joblib"

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    test_data = pd.read_csv(
        test_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True
    )

    X_test = test_data[FEATURE_NAMES].copy()
    y_test = test_data["label"].map(get_family)

    family_model = joblib.load(model_path)
    predictions = family_model.predict(X_test)

    print("Independent test records:", len(X_test))
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    labels = sorted(set(y_test) | set(predictions))

    print("\nConfusion matrix (actual rows, predicted columns):")
    print("Labels:", labels)
    print(confusion_matrix(y_test, predictions, labels=labels))

    print(
        "\nReview the precision and recall for R2L and U2R carefully; "
        "these classes are rare."
    )


if __name__ == "__main__":
    main()
