
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

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


def load_data(filename):
    path = DATASET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True
    )

    if df.shape[1] != 43:
        raise ValueError(f"Expected 43 columns, got {df.shape[1]}")

    if df.isnull().any().any():
        raise ValueError(f"Missing values found in {filename}")

    return df


def prepare_features(df):
    X = df[FEATURE_NAMES].copy()
    y = (df["label"] != "normal").astype("int8")
    return X, y


def create_preprocessor():
    return ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        ),
        ("numerical", StandardScaler(), NUMERICAL_FEATURES)
    ])


if __name__ == "__main__":
    train_df = load_data("KDDTrain+.txt")
    test_df = load_data("KDDTest+.txt")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("PREPROCESSING COMPLETED")
    print("-" * 40)
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Processed features: {X_train_processed.shape[1]}")
    print(f"Training samples: {X_train_processed.shape[0]:,}")
    print(f"Testing samples: {X_test_processed.shape[0]:,}")
    print("\nTraining labels:")
    print(y_train.value_counts().rename(index={
        0: "Normal", 1: "Attack"
    }))
    print("\nTesting labels:")
    print(y_test.value_counts().rename(index={
        0: "Normal", 1: "Attack"
    }))
