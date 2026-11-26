"""
Shared Training Utilities
------------------------------
"""

import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def print_metrics(prefix: str, metrics: dict):
    print(f"{prefix} \u2014 acc: {metrics['accuracy']:.4f}  precision: {metrics['precision']:.4f}  "
          f"recall: {metrics['recall']:.4f}  f1: {metrics['f1']:.4f}")


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred) [legitimate, phishing]:")
    print(cm)


class BestCheckpointSaver:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.best_f1 = -1.0

    def maybe_save(self, model: torch.nn.Module, val_f1: float) -> bool:
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            torch.save(model.state_dict(), self.save_path)
            return True
        return False


def class_weights_from_labels(labels, num_classes: int = 2) -> torch.Tensor:
    labels = np.asarray(labels)
    counts = np.array([np.sum(labels == c) for c in range(num_classes)], dtype=np.float32)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)