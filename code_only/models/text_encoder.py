"""
Text Encoder for Multimodal Spam/Phishing Detection
Uses a pretrained transformer (default: distilbert-base-uncased) to encode
email/SMS/webpage text into a fixed-size embedding.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", embedding_dim: int = 256,
                 freeze_backbone: bool = False, max_length: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        hidden_size = self.backbone.config.hidden_size
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def tokenize(self, texts: list[str]):
        return self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        return self.projection(pooled)


class TextClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)