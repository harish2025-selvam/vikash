import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_COLUMNS, TARGET_COLUMN, REQUIRED_COLUMNS, DATA_PATH


def load_data(path=None):
    """Load and validate dataset. Falls back to default if invalid."""
    try:
        df = pd.read_csv(path or DATA_PATH)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df.dropna(subset=REQUIRED_COLUMNS)
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        if path:
            # fallback to default
            return load_data(None)
        raise e


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def preprocess(df):
    """Return X, y, scaler."""
    X = df[FEATURE_COLUMNS].copy().astype(float)
    y = df[TARGET_COLUMN].copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler, FEATURE_COLUMNS


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def partition_for_federated(X, y, num_clients=10):
    """Split data into equal client partitions."""
    n = len(X)
    size = n // num_clients
    partitions = []
    for i in range(num_clients):
        start = i * size
        end = start + size if i < num_clients - 1 else n
        partitions.append((X[start:end], y[start:end]))
    return partitions
