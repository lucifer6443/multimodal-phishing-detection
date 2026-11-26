"""
Manifest Builder — for the real dataset structure in use
--------------------------------------------------------------
Produces three kinds of outputs from your actual files:

  1. URL-only manifests (for training the standalone URL MLP classifier)
     -- straight from merged_features_final.xlsx, train/test split here
     since that file doesn't ship a predefined split.

  2. Image-only manifests (for training the standalone image classifier)
     -- just walks the existing train/test/{legitimate,phishing}/ folders;
     Phishing dataset already provides the split, so we don't re-split.

  3. Aligned image+URL manifest (for the two-modality fusion model)
     -- joins screenshot filenames to URL rows by domain. Expect this to
     keep only a modest fraction of samples since the two sources weren't
     collected as pairs -- that's fine, this manifest is only for the
     fusion experiment, not the only training signal.

Usage (run directly, from inside the data/ folder — this avoids pulling in
torch just to build manifests, since that's a pandas/sklearn-only job):
    cd data
    python prepare_manifest.py \
        --url_features_path /path/to/merged_features_final.xlsx \
        --image_root "/path/to/Phishing dataset/image" \
        --output_dir ../manifests
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .domain_utils import domain_from_url, domain_from_filename
    from .feature_columns import FEATURE_COLUMNS
except ImportError:
    # Allows running directly as `python prepare_manifest.py` from inside data/
    from domain_utils import domain_from_url, domain_from_filename
    from feature_columns import FEATURE_COLUMNS


def normalize_label(value) -> int:
    v = str(value).strip().lower()
    if v in {"phishing", "phish", "1", "spam"}:
        return 1
    if v in {"legitimate", "legit", "0", "ham"}:
        return 0
    raise ValueError(f"Unrecognized label value: {value!r}")


def build_url_manifests(url_features_path: str, output_dir: str, val_size: float, test_size: float):
    df = pd.read_excel(url_features_path)
    df["label_id"] = df["label"].apply(normalize_label)
    df["domain"] = df["url"].apply(domain_from_url)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns not found: {missing_cols}")

    keep_cols = ["url", "domain"] + FEATURE_COLUMNS + ["label_id"]
    df = df[keep_cols]

    before_count = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    dropped = before_count - len(df)
    if dropped > 0:
        print(f"[URL] Dropped {dropped} rows with missing feature values.")

    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label_id"], random_state=42
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["label_id"], random_state=42
    )

    train_df.to_csv(os.path.join(output_dir, "url_train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "url_val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "url_test_manifest.csv"), index=False)

    print(f"[URL] train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    return df


def build_image_manifests(image_root: str, output_dir: str):
    """Walks image_root/{train,test}/{legitimate,phishing}/*.png"""
    rows = []
    for split in ["train", "test"]:
        split_dir = os.path.join(image_root, split)
        if not os.path.isdir(split_dir):
            print(f"[Image] WARNING: {split_dir} not found, skipping.")
            continue
        for class_name, label_id in [("legitimate", 0), ("phishing", 1)]:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                rows.append(
                    {
                        "image_path": os.path.join(class_dir, fname),
                        "filename": fname,
                        "domain": domain_from_filename(fname),
                        "label_id": label_id,
                        "split": split,
                    }
                )

    df = pd.DataFrame(rows)
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    train_df.to_csv(os.path.join(output_dir, "image_train_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "image_test_manifest.csv"), index=False)

    print(f"[Image] train: {len(train_df)}  test: {len(test_df)}")
    return df


def build_aligned_manifest(image_df: pd.DataFrame, url_df: pd.DataFrame, output_dir: str):
    """Joins on domain. Keeps image split (train/test) as given by the
    image dataset; url features come along for whichever rows match."""
    merged = image_df.merge(
        url_df.drop(columns=["url"]), on="domain", how="inner", suffixes=("_img", "_url")
    )

    # Prefer the image label as ground truth when both exist (image folder
    # labels are the more reliable/curated split in this dataset); flag
    # disagreements instead of silently picking one, so you can inspect them.
    disagreements = merged[merged["label_id_img"] != merged["label_id_url"]]
    if len(disagreements) > 0:
        print(f"[Aligned] WARNING: {len(disagreements)} matched rows have disagreeing labels between "
              f"image and URL source — kept image label, inspect manifests/aligned_label_disagreements.csv")
        disagreements.to_csv(os.path.join(output_dir, "aligned_label_disagreements.csv"), index=False)

    merged = merged.rename(columns={"label_id_img": "label_id"}).drop(columns=["label_id_url"])

    merged.to_csv(os.path.join(output_dir, "aligned_image_url_manifest.csv"), index=False)
    print(f"[Aligned] matched {len(merged)} image+URL pairs by domain "
          f"(out of {len(image_df)} images / {len(url_df)} URL rows)")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_features_path", required=True, help="Path to merged_features_final.xlsx")
    parser.add_argument("--image_root", required=True, help='Path to "Phishing dataset/image" folder')
    parser.add_argument("--output_dir", default="./manifests")
    parser.add_argument("--val_size", type=float, default=0.15, help="For URL-only split (image already has train/test)")
    parser.add_argument("--test_size", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    url_df = build_url_manifests(args.url_features_path, args.output_dir, args.val_size, args.test_size)
    image_df = build_image_manifests(args.image_root, args.output_dir)
    build_aligned_manifest(image_df, url_df, args.output_dir)
