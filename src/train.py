
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

COLUMNS = FEATURE_NAMES + ["label", "difficulty"]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERICAL_FEATURES = [
    col for col in FEATURE_NAMES
    if col not in CATEGORICAL_FEATURES
]


def load_dataset(filename):
    path = DATASET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True
    )

    X = df[FEATURE_NAMES].copy()
    y = (df["label"] != "normal").astype("int8")

    return X, y


def build_preprocessor():
    return ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        ),
        ("numerical", StandardScaler(), NUMERICAL_FEATURES)
    ])


def main():
    print("Loading NSL-KDD dataset...")
    X_train, y_train = load_dataset("KDDTrain+.txt")
    X_test, y_test = load_dataset("KDDTest+.txt")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    }

    results = []

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "Recall (%)": round(recall * 100, 2),
            "F1 Score (%)": round(f1 * 100, 2),
            "Training Time (s)": round(training_time, 2)
        })

        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / model_filename)

        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1 Score:  {f1 * 100:.2f}%")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Saved model: models/{model_filename}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        PROJECT_DIR / "model_results.csv",
        index=False
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("\nResults saved to model_results.csv")
    print("Trained models saved in the models folder.")


if __name__ == "__main__":
    main()
