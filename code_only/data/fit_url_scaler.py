"""
Fit a StandardScaler on the URL training-split features and save it, so
the exact same scaling is reused for val/test/inference (fitting on the
full dataset, or separately per split, would leak information / cause
train-test mismatch).

Usage (run directly, from inside the data/ folder):
    cd data
    python fit_url_scaler.py --train_manifest ../manifests/url_train_manifest.csv --output_path ../manifests/url_scaler.pkl
"""

import argparse
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .feature_columns import FEATURE_COLUMNS
except ImportError:
    from feature_columns import FEATURE_COLUMNS


def fit_and_save_scaler(train_manifest_path: str, output_path: str):
    df = pd.read_csv(train_manifest_path)
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS].values)

    with open(output_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Fitted StandardScaler on {len(df)} rows, saved to {output_path}")
    print("Feature means:", dict(zip(FEATURE_COLUMNS, scaler.mean_.round(3))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--output_path", default="./manifests/url_scaler.pkl")
    args = parser.parse_args()

    fit_and_save_scaler(args.train_manifest, args.output_path)
