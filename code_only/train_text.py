"""
Train the Text Branch - Email Spam/Phishing Classifier
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.text_encoder import TextEncoder, TextClassifierHead
from data.datasets import EmailTextDataset
from data.collate import build_email_text_collate
from training_utils import (
    set_seed, get_device, compute_metrics, print_metrics, print_confusion_matrix,
    BestCheckpointSaver,
)


def run_epoch(model_parts, loader, device, optimizer=None):
    text_encoder, text_head = model_parts
    is_train = optimizer is not None
    text_encoder.train(is_train)
    text_head.train(is_train)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["text_input_ids"].to(device)
        attention_mask = batch["text_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            embedding = text_encoder(input_ids, attention_mask)
            logits = text_head(embedding)
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
    parser.add_argument("--csv_paths", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional: cap dataset size for faster runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    full_dataset = EmailTextDataset(csv_paths=args.csv_paths)
    if args.max_samples is not None and args.max_samples < len(full_dataset):
        full_dataset.df = full_dataset.df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled dataset to {args.max_samples} rows for faster preliminary training.")
    labels = full_dataset.df["category_text"].apply(
        lambda v: 1 if str(v).strip().lower() in {"spam", "phishing", "1"} else 0
    ).tolist()
    indices = list(range(len(full_dataset)))

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=args.seed)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=args.seed)

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    text_encoder = TextEncoder().to(device)
    text_head = TextClassifierHead().to(device)
    collate_fn = build_email_text_collate(text_encoder)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    params = list(text_encoder.parameters()) + list(text_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    saver = BestCheckpointSaver(os.path.join(args.output_dir, "text_model_best.pt"))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch((text_encoder, text_head), train_loader, device, optimizer)
        val_loss, val_metrics, _, _ = run_epoch((text_encoder, text_head), val_loader, device, optimizer=None)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print_metrics("  Train", train_metrics)
        print(f"  Val loss:   {val_loss:.4f}")
        print_metrics("  Val  ", val_metrics)

        improved = saver.maybe_save(
            nn.ModuleDict({"text_encoder": text_encoder, "text_head": text_head}), val_metrics["f1"]
        )
        if improved:
            print(f"  -> New best val F1: {val_metrics['f1']:.4f}, checkpoint saved.")

    best_state = torch.load(os.path.join(args.output_dir, "text_model_best.pt"), map_location=device)
    combined = nn.ModuleDict({"text_encoder": text_encoder, "text_head": text_head})
    combined.load_state_dict(best_state)

    test_loss, test_metrics, test_labels, test_preds = run_epoch(
        (text_encoder, text_head), test_loader, device, optimizer=None
    )
    print(f"\n=== Final Test Results (best checkpoint) ===")
    print(f"Test loss: {test_loss:.4f}")
    print_metrics("Test", test_metrics)
    print_confusion_matrix(test_labels, test_preds)


if __name__ == "__main__":
    main()