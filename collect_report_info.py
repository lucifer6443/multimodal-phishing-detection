"""
Collect the missing numbers for the project report.

Run from the project root (C:\\Users\\Aarish\\p360project):
    python collect_report_info.py

Prints everything and also saves it to report_info.txt.
Nothing here trains anything; it only reads your existing files and
evaluates the saved URL checkpoint on the URL test set (takes seconds).
"""

import os
import pickle
import platform
import subprocess
import sys
import time

import pandas as pd

OUT_LINES = []


def log(text=""):
    print(text)
    OUT_LINES.append(str(text))


def section(title):
    log("")
    log("=" * 60)
    log(title)
    log("=" * 60)


DATASET_ROOT = r"C:\Users\Aarish\Documents\Dataset"
EMAIL_CSVS = [
    "final_cleaned_safe_1.csv",
    "final_cleaned_phis1.csv",
    "final_cleaned_phis2.csv",
    "final_cleaned_phis_3.csv",
    "final_cleaned_phis_4.csv",
]


def find_file(root, filename, max_depth=4):
    """Walk root up to max_depth levels, skipping big image folders."""
    root_depth = root.rstrip("\\/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - root_depth
        dirnames[:] = [d for d in dirnames if d.lower() != "image"]
        if depth >= max_depth:
            dirnames[:] = []
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


# ---------------------------------------------------------------- 1
section("1. URL data: rows dropped for missing values + split sizes")
try:
    from data.feature_columns import FEATURE_COLUMNS

    xlsx_path = find_file(DATASET_ROOT, "merged_features_final.xlsx")
    if xlsx_path is None:
        log("merged_features_final.xlsx not found under " + DATASET_ROOT)
    else:
        log(f"Found: {xlsx_path}")
        raw = pd.read_excel(xlsx_path)
        log(f"Total rows in the Excel file: {len(raw)}")
        missing_per_col = raw[FEATURE_COLUMNS].isna().sum()
        missing_per_col = missing_per_col[missing_per_col > 0]
        log("Columns with missing values (count of missing cells):")
        log(missing_per_col.to_string() if len(missing_per_col) else "  none")
        rows_with_missing = int(raw[FEATURE_COLUMNS].isna().any(axis=1).sum())
        log(f"Rows dropped for missing values: {rows_with_missing}")
        log(f"Rows kept: {len(raw) - rows_with_missing}")
except Exception as e:
    log(f"Could not compute dropped rows: {e!r}")

for name in [
    "url_train_manifest.csv",
    "url_val_manifest.csv",
    "url_test_manifest.csv",
    "image_train_manifest.csv",
    "image_test_manifest.csv",
    "aligned_image_url_manifest.csv",
]:
    path = os.path.join("manifests", name)
    try:
        df = pd.read_csv(path)
        counts = df["label_id"].value_counts().to_dict()
        log(f"{name}: {len(df)} rows  (label_id counts: {counts})")
    except Exception as e:
        log(f"{name}: could not read ({e!r})")

# ---------------------------------------------------------------- 2
section("2. URL test confusion matrix (best saved checkpoint)")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from models.url_feature_encoder import URLFeatureEncoder, URLFeatureClassifierHead
    from data.datasets import URLFeatureDataset
    from data.collate import url_feature_collate_fn
    from training_utils import get_device, print_metrics, print_confusion_matrix
    from train_url import run_epoch

    device = get_device()
    with open("manifests/url_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    test_set = URLFeatureDataset("manifests/url_test_manifest.csv", scaler=scaler)
    loader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=url_feature_collate_fn)

    encoder = URLFeatureEncoder(num_features=18).to(device)
    head = URLFeatureClassifierHead().to(device)
    combined = nn.ModuleDict({"url_encoder": encoder, "head": head})
    combined.load_state_dict(torch.load("checkpoints/url_model_best.pt", map_location=device))

    _, metrics, labels, preds = run_epoch((encoder, head), loader, device, class_weights=None, optimizer=None)
    log(f"Test samples: {len(test_set)}")
    print_metrics("URL test", metrics)
    print_confusion_matrix(labels, preds)
    log("(rows = true class, columns = predicted class, order = [legitimate, phishing])")
    OUT_LINES.append(f"URL metrics: {metrics}")
except Exception as e:
    log(f"Could not evaluate URL checkpoint: {e!r}")

# ---------------------------------------------------------------- 3
section("3. Laptop CPU and RAM, software versions")
try:
    cpu = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Processor | Select-Object -First 1 | "
         "ForEach-Object { $_.Name + ' | cores=' + $_.NumberOfCores + ' | threads=' + $_.NumberOfLogicalProcessors }"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    ram = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    log(f"CPU: {cpu}")
    log(f"RAM: {ram} GB")
except Exception as e:
    log(f"Could not read hardware info: {e!r}")
log(f"OS: {platform.platform()}")
log(f"Python: {sys.version.split()[0]}")
try:
    import torch
    log(f"PyTorch: {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
except Exception:
    pass

# ---------------------------------------------------------------- 4
section("4. Saved checkpoints (files only; see note on image epoch)")
for ck in ["image_model_best.pt", "url_model_best.pt", "text_model_best.pt", "fusion_model_best.pt"]:
    path = os.path.join("checkpoints", ck)
    if os.path.exists(path):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        log(f"{ck}: {os.path.getsize(path) / 1e6:.1f} MB, last saved {stamp}")
    else:
        log(f"{ck}: not found")
log("Note: the checkpoint stores only weights, not the epoch number. The epoch")
log("is only in the training log line '-> New best val F1 ... checkpoint saved.'")

# ---------------------------------------------------------------- 5
section("5. Email CSVs: row counts and any readme/licence files")
total = 0
for name in EMAIL_CSVS:
    path = os.path.join(DATASET_ROOT, name)
    try:
        df = pd.read_csv(path)
        total += len(df)
        extra = ""
        if "category_text" in df.columns:
            extra = "  labels: " + str(df["category_text"].value_counts().to_dict())
        log(f"{name}: {len(df)} rows{extra}")
    except Exception as e:
        log(f"{name}: could not read ({e!r})")
log(f"Total email rows across the five CSVs: {total}")

try:
    notes = [f for f in os.listdir(DATASET_ROOT)
             if f.lower().endswith((".txt", ".md", ".pdf", ".docx", ".url", ".html"))
             or "readme" in f.lower() or "licen" in f.lower()]
    log("Readme / licence / note files at the top of the Dataset folder: "
        + (", ".join(notes) if notes else "none"))
except Exception as e:
    log(f"Could not list Dataset folder: {e!r}")

with open("report_info.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT_LINES))
log("")
log("Saved to report_info.txt")
