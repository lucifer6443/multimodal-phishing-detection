"""
Evaluate a Saved Image Checkpoint on the Test Set
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.image_encoder import ImageEncoder, ImageClassifierHead, get_recommended_transform
from data.datasets import ImageOnlyDataset
from data.collate import image_only_collate_fn
from training_utils import get_device, compute_metrics, print_metrics, print_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b_16"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    transform = get_recommended_transform(backbone=args.backbone)
    test_dataset = ImageOnlyDataset(args.test_manifest, image_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=image_only_collate_fn)
    print(f"Test samples: {len(test_dataset)}")

    image_encoder = ImageEncoder(backbone=args.backbone).to(device)
    head = ImageClassifierHead().to(device)
    combined = nn.ModuleDict({"image_encoder": image_encoder, "head": head})

    state = torch.load(args.checkpoint, map_location=device)
    combined.load_state_dict(state)
    print(f"Loaded checkpoint from {args.checkpoint}")

    image_encoder.eval()
    head.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            embedding = image_encoder(images)
            logits = head(embedding)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(test_dataset)
    metrics = compute_metrics(all_labels, all_preds)

    print(f"\n=== Test Results (loaded checkpoint) ===")
    print(f"Test loss: {avg_loss:.4f}")
    print_metrics("Test", metrics)
    print_confusion_matrix(all_labels, all_preds)


if __name__ == "__main__":
    main()