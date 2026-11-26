"""
End-to-End Data Loading Demo — real dataset structure
------------------------------------------------------------
Run after:
  1. python data/prepare_manifest.py --url_features_path ... --image_root ...
  2. python data/fit_url_scaler.py --train_manifest manifests/url_train_manifest.csv
  3. pip install -r requirements.txt

Wires each manifest into its DataLoader and confirms shapes through each
of the three classifiers (text, image, URL) plus the image+URL fusion
model, mirroring the standalone smoke tests in each model file.
"""

import pickle
import torch
from torch.utils.data import DataLoader

from models.text_encoder import TextEncoder, TextClassifierHead
from models.image_encoder import ImageEncoder, get_recommended_transform
from models.url_feature_encoder import URLFeatureEncoder, URLFeatureClassifierHead
from models.fusion_model import ImageURLFusionDetector

from data.datasets import EmailTextDataset, ImageOnlyDataset, URLFeatureDataset, AlignedImageURLDataset
from data.collate import (
    build_email_text_collate,
    image_only_collate_fn,
    url_feature_collate_fn,
    aligned_image_url_collate_fn,
)


def main():
    image_transform = get_recommended_transform(backbone="resnet50")

    # --- 1. Text branch ---
    text_encoder = TextEncoder()
    text_head = TextClassifierHead()
    text_dataset = EmailTextDataset(
        csv_paths=[
            r"C:\Users\Aarish\Documents\Dataset\final_cleaned_safe_1.csv",
            r"C:\Users\Aarish\Documents\Dataset\final_cleaned_phis1.csv",
            r"C:\Users\Aarish\Documents\Dataset\final_cleaned_phis2.csv",
            r"C:\Users\Aarish\Documents\Dataset\final_cleaned_phis_3.csv",
            r"C:\Users\Aarish\Documents\Dataset\final_cleaned_phis_4.csv",
        ]
    )
    text_loader = DataLoader(
        text_dataset, batch_size=32, shuffle=True, collate_fn=build_email_text_collate(text_encoder)
    )
    text_batch = next(iter(text_loader))
    text_emb = text_encoder(text_batch["text_input_ids"], text_batch["text_attention_mask"])
    text_logits = text_head(text_emb)
    print("Text branch -> logits shape:", text_logits.shape)

    # --- 2. Image branch ---
    image_encoder = ImageEncoder(backbone="resnet50")
    image_dataset = ImageOnlyDataset("manifests/image_train_manifest.csv", image_transform=image_transform)
    image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True, collate_fn=image_only_collate_fn)
    image_batch = next(iter(image_loader))
    image_emb = image_encoder(image_batch["images"])
    print("Image branch -> embedding shape:", image_emb.shape)

    # --- 3. URL branch ---
    with open("manifests/url_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    url_encoder = URLFeatureEncoder(num_features=18)
    url_head = URLFeatureClassifierHead()
    url_dataset = URLFeatureDataset("manifests/url_train_manifest.csv", scaler=scaler)
    url_loader = DataLoader(url_dataset, batch_size=64, shuffle=True, collate_fn=url_feature_collate_fn)
    url_batch = next(iter(url_loader))
    url_emb = url_encoder(url_batch["url_features"])
    url_logits = url_head(url_emb)
    print("URL branch -> logits shape:", url_logits.shape)

    # --- 4. Image + URL fusion ---
    fusion_model = ImageURLFusionDetector()
    aligned_dataset = AlignedImageURLDataset(
        "manifests/aligned_image_url_manifest.csv", image_transform=image_transform, scaler=scaler
    )
    aligned_loader = DataLoader(
        aligned_dataset, batch_size=16, shuffle=True, collate_fn=aligned_image_url_collate_fn
    )
    aligned_batch = next(iter(aligned_loader))
    fusion_logits = fusion_model(aligned_batch["images"], aligned_batch["url_features"])
    print("Fusion (image+URL) -> logits shape:", fusion_logits.shape)


if __name__ == "__main__":
    main()
