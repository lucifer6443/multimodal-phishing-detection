"""
Train the Image Branch - Screenshot Classifier
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.image_encoder import ImageEncoder, ImageClassifierHead, get_recommended_transform
from data.datasets import ImageOnlyDataset
from data.collate import image_only_collate_fn
from training_utils import (
    set_seed, get_device, compute_metrics, print_metrics, print_confusion_matrix,
    BestCheckpointSaver,
)


def run_epoch(model_parts, loader, device, optimizer=None):
    image_encoder, head = model_parts
    is_train = optimizer is not None
    image_encoder.train(is_train)
    head.train(is_train)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            embedding = image_encoder(images)
            logits = head(embedding)
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
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit_b_16"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    transform = get_recommended_transform(backbone=args.backbone)
    full_train_dataset = ImageOnlyDataset(args.train_manifest, image_transform=transform)
    test_dataset = ImageOnlyDataset(args.test_manifest, image_transform=transform)

    labels = full_train_dataset.df["label_id"].tolist()
    indices = list(range(len(full_train_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, stratify=labels, random_state=args.seed)

    train_set = Subset(full_train_dataset, train_idx)
    val_set = Subset(full_train_dataset, val_idx)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_dataset)}")

    image_encoder = ImageEncoder(backbone=args.backbone).to(device)
    head = ImageClassifierHead().to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=image_only_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=image_only_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=image_only_collate_fn)

    params = list(image_encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    saver = BestCheckpointSaver(os.path.join(args.output_dir, "image_model_best.pt"))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch((image_encoder, head), train_loader, device, optimizer)
        val_loss, val_metrics, _, _ = run_epoch((image_encoder, head), val_loader, device, optimizer=None)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print_metrics("  Train", train_metrics)
        print(f"  Val loss:   {val_loss:.4f}")
        print_metrics("  Val  ", val_metrics)

        improved = saver.maybe_save(
            nn.ModuleDict({"image_encoder": image_encoder, "head": head}), val_metrics["f1"]
        )
        if improved:
            print(f"  -> New best val F1: {val_metrics['f1']:.4f}, checkpoint saved.")

    best_state = torch.load(os.path.join(args.output_dir, "image_model_best.pt"), map_location=device)
    combined = nn.ModuleDict({"image_encoder": image_encoder, "head": head})
    combined.load_state_dict(best_state)

    test_loss, test_metrics, test_labels, test_preds = run_epoch(
        (image_encoder, head), test_loader, device, optimizer=None
    )
    print(f"\n=== Final Test Results (best checkpoint) ===")
    print(f"Test loss: {test_loss:.4f}")
    print_metrics("Test", test_metrics)
    print_confusion_matrix(test_labels, test_preds)


if __name__ == "__main__":
    main()