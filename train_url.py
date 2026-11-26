"""
Train the URL Branch - Engineered-Feature Classifier
"""

import argparse
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.url_feature_encoder import URLFeatureEncoder, URLFeatureClassifierHead
from data.datasets import URLFeatureDataset
from data.collate import url_feature_collate_fn
from training_utils import (
    set_seed, get_device, compute_metrics, print_metrics, print_confusion_matrix,
    BestCheckpointSaver, class_weights_from_labels,
)


def run_epoch(model_parts, loader, device, class_weights=None, optimizer=None):
    url_encoder, head = model_parts
    is_train = optimizer is not None
    url_encoder.train(is_train)
    head.train(is_train)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        features = batch["url_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            embedding = url_encoder(features)
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
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--scaler_path", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    train_set = URLFeatureDataset(args.train_manifest, scaler=scaler)
    val_set = URLFeatureDataset(args.val_manifest, scaler=scaler)
    test_set = URLFeatureDataset(args.test_manifest, scaler=scaler)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    class_weights = class_weights_from_labels(train_set.df["label_id"].tolist())
    print(f"Class weights (legitimate, phishing): {class_weights.tolist()}")

    url_encoder = URLFeatureEncoder(num_features=18).to(device)
    head = URLFeatureClassifierHead().to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=url_feature_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=url_feature_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=url_feature_collate_fn)

    params = list(url_encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    saver = BestCheckpointSaver(os.path.join(args.output_dir, "url_model_best.pt"))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            (url_encoder, head), train_loader, device, class_weights, optimizer
        )
        val_loss, val_metrics, _, _ = run_epoch(
            (url_encoder, head), val_loader, device, class_weights, optimizer=None
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print_metrics("  Train", train_metrics)
        print(f"  Val loss:   {val_loss:.4f}")
        print_metrics("  Val  ", val_metrics)

        improved = saver.maybe_save(
            nn.ModuleDict({"url_encoder": url_encoder, "head": head}), val_metrics["f1"]
        )
        if improved:
            print(f"  -> New best val F1: {val_metrics['f1']:.4f}, checkpoint saved.")

    best_state = torch.load(os.path.join(args.output_dir, "url_model_best.pt"), map_location=device)
    combined = nn.ModuleDict({"url_encoder": url_encoder, "head": head})
    combined.load_state_dict(best_state)

    test_loss, test_metrics, test_labels, test_preds = run_epoch(
        (url_encoder, head), test_loader, device, class_weights=None, optimizer=None
    )
    print(f"\n=== Final Test Results (best checkpoint) ===")
    print(f"Test loss: {test_loss:.4f}")
    print_metrics("Test", test_metrics)
    print_confusion_matrix(test_labels, test_preds)


if __name__ == "__main__":
    main()