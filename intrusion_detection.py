import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target", "difficulty_level"
]

TRAIN_PATH = "data/KDDTrain+.txt"
TEST_PATH = "data/KDDTest+.txt"
MODEL_PATH = "model.joblib"


def load_data(path):
    if not os.path.exists(path):
        print(f"Error: dataset not found at '{path}'")
        sys.exit(1)
    df = pd.read_csv(path, names=columns, header=None)
    return df


def preprocess(df, fit_encoder=None):
    df = df.copy()
    df["label"] = df["target"].apply(lambda x: 0 if x == "normal" else 1)
    X = df.drop(["target", "difficulty_level", "label"], axis=1)
    y = df["label"]
    X = pd.get_dummies(X)
    return X, y


def align_columns(X_train, X_test):
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    extra_cols = set(X_test.columns) - set(X_train.columns)
    X_test = X_test.drop(columns=extra_cols)
    X_test = X_test[X_train.columns]
    return X_test


def print_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 50)
    for i in range(min(top_n, len(indices))):
        print(f"{i+1:2d}. {feature_names[indices[i]]:35s} ({importances[indices[i]]:.4f})")


def main():
    print("Loading training data...")
    df_train = load_data(TRAIN_PATH)
    X_train, y_train = preprocess(df_train)
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    print("\nLoading test data...")
    df_test = load_data(TEST_PATH)
    X_test, y_test = preprocess(df_test)
    X_test = align_columns(X_train, X_test)
    print(f"Test samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")

    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out test set...")
    predictions = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Normal", "Attack"]))

    print_feature_importance(model, X_train.columns)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
