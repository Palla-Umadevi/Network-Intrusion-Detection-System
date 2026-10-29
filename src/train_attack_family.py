
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

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
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERICAL = [c for c in FEATURE_NAMES if c not in CATEGORICAL]

# NSL-KDD attack labels grouped into their standard families.
ATTACK_FAMILIES = {
    "dos": {
        "back", "land", "neptune", "pod", "smurf", "teardrop",
        "apache2", "udpstorm", "processtable", "worm", "mailbomb"
    },
    "probe": {
        "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"
    },
    "r2l": {
        "ftp_write", "guess_passwd", "imap", "multihop", "phf",
        "spy", "warezclient", "warezmaster", "xlock", "xsnoop",
        "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
        "named"
    },
    "u2r": {
        "buffer_overflow", "loadmodule", "perl", "rootkit",
        "ps", "sqlattack", "xterm"
    }
}


def get_family(label):
    label = str(label).strip().lower().rstrip(".")

    if label == "normal":
        return "Normal"

    for family, attacks in ATTACK_FAMILIES.items():
        if label in attacks:
            return family.upper()

    # Keep unrecognized labels separate rather than inventing a family.
    return "Other/Unmapped"


def main():
    path = DATASET_DIR / "KDDTrain+.txt"

    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    data = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True
    )

    X = data[FEATURE_NAMES].copy()
    y = data["label"].map(get_family)

    print("Training records:", len(X))
    print("\nFamily distribution:")
    print(y.value_counts().to_string())

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL
        ),
        ("numerical", StandardScaler(), NUMERICAL)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    print("\nTraining attack-family classifier...")
    pipeline.fit(X, y)

    output = MODEL_DIR / "attack_family.joblib"
    joblib.dump(pipeline, output)

    print(f"\nSaved model: {output}")
    print("\nTraining-set report (not independent test performance):")
    print(classification_report(
        y, pipeline.predict(X), zero_division=0
    ))
    print(
        "\nImportant: evaluate on KDDTest+ before treating "
        "these predictions as reliable."
    )


if __name__ == "__main__":
    main()
