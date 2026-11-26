"""
Train the Fusion Branch - Image + URL Combined Classifier
"""

import argparse
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.fusion_model import ImageURLFusionDetector
from models.image_encoder import get_recommended_transform
from data.datasets import AlignedImageURLDataset
from data.collate import aligned_image_url_collate_fn
from training_utils import (
    set_seed, get_device, compute_metrics, print_metrics, print_confusion_matrix,
    BestCheckpointSaver, class_weights_from_labels,
)


def run_epoch(model, loader, device, class_weights=None, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["images"].to(device)
        url_features = batch["url_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images, url_features)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_manifest", required=True)
    parser.add_argument("--scaler_path", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    transform = get_recommended_transform(backbone=args.backbone)
    full_dataset = AlignedImageURLDataset(args.aligned_manifest, image_transform=transform, scaler=scaler)

    labels = full_dataset.df["label_id"].tolist()
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=args.seed)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=args.seed)

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_labels = [labels[i] for i in train_idx]
    class_weights = class_weights_from_labels(train_labels)
    print(f"Class weights (legitimate, phishing): {class_weights.tolist()}")

    model = ImageURLFusionDetector(image_backbone=args.backbone).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=aligned_image_url_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=aligned_image_url_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=aligned_image_url_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    saver = BestCheckpointSaver(os.path.join(args.output_dir, "fusion_model_best.pt"))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(model, train_loader, device, class_weights, optimizer)
        val_loss, val_metrics, _, _ = run_epoch(model, val_loader, device, class_weights, optimizer=None)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print_metrics("  Train", train_metrics)
        print(f"  Val loss:   {val_loss:.4f}")
        print_metrics("  Val  ", val_metrics)

        improved = saver.maybe_save(model, val_metrics["f1"])
        if improved:
            print(f"  -> New best val F1: {val_metrics['f1']:.4f}, checkpoint saved.")

    best_state = torch.load(os.path.join(args.output_dir, "fusion_model_best.pt"), map_location=device)
    model.load_state_dict(best_state)

    test_loss, test_metrics, test_labels, test_preds = run_epoch(
        model, test_loader, device, class_weights=None, optimizer=None
    )
    print(f"\n=== Final Test Results (best checkpoint) ===")
    print(f"Test loss: {test_loss:.4f}")
    print_metrics("Test", test_metrics)
    print_confusion_matrix(test_labels, test_preds)


if __name__ == "__main__":
    main()