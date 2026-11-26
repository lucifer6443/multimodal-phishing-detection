"""
Collate Functions — matching the real datasets
----------------------------------------------------
"""

from functools import partial
import torch


def email_text_collate_fn(batch: list[dict], text_encoder) -> dict:
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    text_batch = text_encoder.tokenize(texts)
    return {
        "text_input_ids": text_batch["input_ids"],
        "text_attention_mask": text_batch["attention_mask"],
        "labels": labels,
    }


def image_only_collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"images": images, "labels": labels}


def url_feature_collate_fn(batch: list[dict]) -> dict:
    features = torch.tensor([item["url_features"] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"url_features": features, "labels": labels}


def aligned_image_url_collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    features = torch.tensor([item["url_features"] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"images": images, "url_features": features, "labels": labels}


def build_email_text_collate(text_encoder):
    return partial(email_text_collate_fn, text_encoder=text_encoder)
