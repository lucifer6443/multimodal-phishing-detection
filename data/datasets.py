"""
PyTorch Datasets — matching the actual data in use
--------------------------------------------------------
Four dataset classes, one per manifest produced by prepare_manifest.py
(plus the raw email CSVs, which need no manifest step):

  1. EmailTextDataset      — final_cleaned_safe_1.csv + final_cleaned_phis1-4.csv
  2. ImageOnlyDataset      — image_{train,test}_manifest.csv
  3. URLFeatureDataset     — url_{train,val,test}_manifest.csv
  4. AlignedImageURLDataset — aligned_image_url_manifest.csv (fusion model)
"""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

try:
    from .feature_columns import FEATURE_COLUMNS
except ImportError:
    from feature_columns import FEATURE_COLUMNS


class EmailTextDataset(Dataset):
    """Combine final_cleaned_safe_1.csv (ham) and final_cleaned_phis1-4.csv
    (spam/phishing) yourself with pandas.concat before passing a single
    CSV path here, or pass a list of paths and this class will concatenate
    them. Columns expected: body, subject, category_text (ham/spam or
    legitimate/phishing).
    """

    def __init__(self, csv_paths, text_col: str = "body", label_col: str = "category_text"):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        frames = [pd.read_csv(p) for p in csv_paths]
        self.df = pd.concat(frames, ignore_index=True)
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row.get("subject", "")) + ". " + str(row[self.text_col])
        label_raw = str(row[self.label_col]).strip().lower()
        label = 1 if label_raw in {"spam", "phishing", "1"} else 0
        return {"text": text, "label": label}


class ImageOnlyDataset(Dataset):
    """Reads image_{train,test}_manifest.csv: columns image_path, label_id."""

    def __init__(self, manifest_path: str, image_transform=None):
        self.df = pd.read_csv(manifest_path)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        return {"image": image, "label": int(row["label_id"])}


class URLFeatureDataset(Dataset):
    """Reads url_{train,val,test}_manifest.csv. Standardization (zero mean,
    unit variance) should be fit on the TRAIN split only and reused for
    val/test — pass a fitted `scaler` (sklearn StandardScaler) here, or
    None to get raw feature values (e.g. while fitting the scaler itself).
    """

    def __init__(self, manifest_path: str, scaler=None):
        self.df = pd.read_csv(manifest_path)
        self.scaler = scaler
        self.raw_features = self.df[FEATURE_COLUMNS].values.astype("float32")
        self.features = self.scaler.transform(self.raw_features) if self.scaler else self.raw_features

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "url_features": self.features[idx],
            "label": int(self.df.iloc[idx]["label_id"]),
        }


class AlignedImageURLDataset(Dataset):
    """Reads aligned_image_url_manifest.csv (image_path + FEATURE_COLUMNS +
    label_id, produced by joining screenshots to URL rows on domain)."""

    def __init__(self, manifest_path: str, image_transform=None, scaler=None):
        self.df = pd.read_csv(manifest_path)
        self.image_transform = image_transform
        self.scaler = scaler
        self.raw_features = self.df[FEATURE_COLUMNS].values.astype("float32")
        self.features = self.scaler.transform(self.raw_features) if self.scaler else self.raw_features

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        return {
            "image": image,
            "url_features": self.features[idx],
            "label": int(row["label_id"]),
        }
