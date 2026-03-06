

import os
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight


def gnn_training():

    # =====================================================
    # PATH CONFIG
    # =====================================================
    DATA_PATH = "media/data"
    MEDIA_PATH = "media"
    RESULT_PATH = "models"

    os.makedirs(MEDIA_PATH, exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)

    # =====================================================
    # DATASET CONFIG
    # =====================================================
    DATASETS = {
        "decimal_benign.csv": 0,
        "decimal_DoS.csv": 1,
        "decimal_spoofing-GAS.csv": 2,
        "decimal_spoofing-RPM.csv": 3,
        "decimal_spoofing-SPEED.csv": 4,
        "decimal_spoofing-STEERING_WHEEL.csv": 5
    }

    # =====================================================
    # LOAD & MERGE DATA
    # =====================================================
    dfs = []

    for filename, label in DATASETS.items():
        file_path = os.path.join(DATA_PATH, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{filename} not found")

        df = pd.read_csv(file_path)
        df["label"] = label
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print("Merged Dataset Shape:", data.shape)

    # =====================================================
    # CLEAN DATA
    # =====================================================
    data.dropna(inplace=True)
    data = data.select_dtypes(include=[np.number])
    data.reset_index(drop=True, inplace=True)
    print("Cleaned Dataset Shape:", data.shape)

    # =====================================================
    # GRAPH FEATURE EXTRACTION
    # =====================================================
    WINDOW = 2

    def extract_graph_features(series):
        return pd.DataFrame({
        "mean": series.rolling(WINDOW, min_periods=WINDOW).mean(),
        "std":  series.rolling(WINDOW, min_periods=WINDOW).std(),
        "max":  series.rolling(WINDOW, min_periods=WINDOW).max(),
        "min":  series.rolling(WINDOW, min_periods=WINDOW).min(),
        "diff": series.diff(),
        "abs_diff": series.diff().abs(),
        "rate": series.diff() / (series.shift(1) + 1e-6)
    })

    graph_features = []

    for col in data.columns:
        if col != "label":
            gf = extract_graph_features(data[col])
            gf.columns = [f"{col}_{c}" for c in gf.columns]
            graph_features.append(gf)

    graph_df = pd.concat(graph_features, axis=1)
    final_df = pd.concat([graph_df, data["label"]], axis=1)
    final_df.dropna(inplace=True)

    print("Final Feature Shape:", final_df.shape)

    # =====================================================
    # TRAIN–TEST SPLIT
    # =====================================================
    X = final_df.drop("label", axis=1)
    y = final_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Samples:", X_train.shape)
    print("Testing Samples :", X_test.shape)

    # =====================================================
    # SCALING + SAVE FEATURE SCHEMA
    # =====================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    FEATURE_COLUMNS = X.columns.tolist()

    joblib.dump(scaler, os.path.join(RESULT_PATH, "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(RESULT_PATH, "feature_columns.pkl"))

    print("Scaler & feature schema saved")

    # =====================================================
    # HANDLE CLASS IMBALANCE
    # =====================================================
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # =====================================================
    # TRAIN HGFT-IDS (FINE TREE)
    # =====================================================
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    joblib.dump(model, os.path.join(RESULT_PATH, "hgft_ids_model.pkl"))

    print("Model trained & saved")

    # =====================================================
    # EVALUATION
    # =====================================================
    y_pred = model.predict(X_test_scaled)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-Score :", f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # =====================================================
    # SAVE METRICS
    # =====================================================
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open(os.path.join(MEDIA_PATH, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(MEDIA_PATH, "metrics.csv"), index=False
    )

    print("Metrics saved")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("HGFT-IDS Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MEDIA_PATH, "confusion_matrix.png"))
    plt.close()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), X.columns[indices])
    plt.title("Top 20 Graph-Based Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(MEDIA_PATH, "feature_importance.png"))
    plt.close()

    print("HGFT-IDS training completed successfully")
    return True


