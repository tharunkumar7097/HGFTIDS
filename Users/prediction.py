# =====================================================
# HGFT-IDS PREDICTION (JUPYTER)
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
from collections import deque

# =====================================================
# CONFIGURATION (MUST MATCH TRAINING)
# =====================================================
WINDOW = 2
NUM_CHANNELS = 8

# Sliding window buffer
sensor_buffer = deque(maxlen=WINDOW)

# =====================================================
# LOAD MODEL, SCALER, FEATURE SCHEMA
# =====================================================
BASE_DIR = os.getcwd()   # Jupyter working directory

MODEL_PATH    = os.path.join(BASE_DIR, "media", "hgft_ids_model.pkl")
SCALER_PATH   = os.path.join(BASE_DIR, "media", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "media", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

print("Model, scaler, and feature schema loaded")

# =====================================================
# ATTACK LABELS
# =====================================================
ATTACK_LABELS = {
    0: "BENIGN",
    1: "DoS ATTACK",
    2: "SPOOFING – GAS",
    3: "SPOOFING – RPM",
    4: "SPOOFING – SPEED",
    5: "SPOOFING – STEERING WHEEL"
}

# =====================================================
# GRAPH FEATURE EXTRACTION (WINDOW = 2)
# =====================================================
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

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def hgft_predict(input_values):
    """
    input_values: list of 8 floats [DATA_0 ... DATA_7]
    """

    if len(input_values) != NUM_CHANNELS:
        raise ValueError("Exactly 8 sensor values are required")

    # Add to sliding buffer
    sensor_buffer.append(input_values)

    # Wait until window is full
    if len(sensor_buffer) < WINDOW:
        return f"Collecting data... ({len(sensor_buffer)}/{WINDOW})"

    # Create DataFrame (same as training)
    df = pd.DataFrame(
        list(sensor_buffer),
        columns=[f"DATA_{i}" for i in range(NUM_CHANNELS)]
    )

    # Graph feature extraction
    graph_features = []

    for col in df.columns:
        gf = extract_graph_features(df[col])
        gf.columns = [f"{col}_{c}" for c in gf.columns]
        graph_features.append(gf)

    graph_df = pd.concat(graph_features, axis=1)

    # Use latest timestep
    X = graph_df.tail(1)

    # Align with training feature schema
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_columns]

    # Scale & predict
    X_scaled = scaler.transform(X)
    pred_class = int(model.predict(X_scaled)[0])

    return ATTACK_LABELS.get(pred_class, "UNKNOWN ATTACK")

hgft_predict([1200, 30, 55, 10, 0.5, 1, 0, 300])
hgft_predict([1250, 35, 80, 12, 0.9, 1, 1, 350])
