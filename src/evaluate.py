
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "KDDTest+.txt"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

COLUMNS = FEATURE_NAMES + ["label", "difficulty"]

MODELS = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}


def load_test_data():
    """Load NSL-KDD test data and convert labels to Normal=0, Attack=1."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH, header=None, names=COLUMNS)

    X_test = data[FEATURE_NAMES]
    y_test = (data["label"].str.strip().str.lower() != "normal").astype(int)

    return X_test, y_test


def main():
    X_test, y_test = load_test_data()

    results = []
    confusion_matrices = {}

    print(f"Test records: {len(X_test):,}")
    print(f"Normal records: {(y_test == 0).sum():,}")
    print(f"Attack records: {(y_test == 1).sum():,}")

    for model_name, filename in MODELS.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print(f"\nSkipping {model_name}: model not found at {model_path}")
            continue

        model = joblib.load(model_path)
        predictions = model.predict(X_test)

        # Label order: 0 = Normal, 1 = Attack
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Attack Precision": precision,
            "Attack Recall": recall,
            "Attack F1": f1,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
            "Missed Attack Rate": fn / (fn + tp) if fn + tp else 0,
        })

        confusion_matrices[model_name] = (y_test, predictions)

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )

        report_path = OUTPUT_DIR / (
            model_name.lower().replace(" ", "_") + "_report.txt"
        )
        report_path.write_text(report, encoding="utf-8")

        print(f"\n{'=' * 55}")
        print(model_name)
        print(f"{'=' * 55}")
        print(f"Accuracy:             {accuracy:.2%}")
        print(f"Attack precision:     {precision:.2%}")
        print(f"Attack recall:        {recall:.2%}")
        print(f"Attack F1-score:      {f1:.2%}")
        print(f"False positives:      {fp:,}")
        print(f"False negatives:      {fn:,}")
        print(f"Missed attack rate:   {fn / (fn + tp):.2%}")
        print("\nClassification report:")
        print(report)

    if not results:
        raise RuntimeError("No trained models were found. Train the models first.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Save confusion-matrix charts
    count = len(confusion_matrices)
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 5))
    if count == 1:
        axes = [axes]

    for ax, (model_name, (actual, predicted)) in zip(
        axes, confusion_matrices.items()
    ):
        ConfusionMatrixDisplay.from_predictions(
            actual,
            predicted,
            labels=[0, 1],
            display_labels=["Normal", "Attack"],
            cmap="Blues",
            values_format="d",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(model_name)

    fig.suptitle("NIDS Model Confusion Matrices")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrices.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Compare key metrics across models
    metric_columns = [
        "Accuracy", "Attack Precision", "Attack Recall", "Attack F1"
    ]
    chart_data = results_df.set_index("Model")[metric_columns]

    ax = chart_data.plot(kind="bar", figsize=(11, 6), ylim=(0, 1))
    ax.set_title("NIDS Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "metric_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    # Compare missed attacks separately
    ax = results_df.plot(
        x="Model",
        y="False Negatives",
        kind="bar",
        legend=False,
        figsize=(9, 5),
    )
    ax.set_title("Attacks Missed by Each NIDS Model")
    ax.set_ylabel("Number of missed attack records")
    ax.set_xlabel("Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "false_negatives.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    print("\nEvaluation files saved in:", OUTPUT_DIR)
    print("- model_comparison.csv")
    print("- confusion_matrices.png")
    print("- metric_comparison.png")
    print("- false_negatives.png")
    print("- One classification report per model")


if __name__ == "__main__":
    main()
