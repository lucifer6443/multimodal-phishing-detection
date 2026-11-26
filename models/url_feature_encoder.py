"""
URL Feature Encoder (MLP) — for engineered numeric URL/DNS features
------------------------------------------------------------------------
Unlike a raw-string URL, this dataset ships 18 pre-engineered lexical and
DNS features per URL (entropy, length, subdomain count, TTL, name server
count, etc.) rather than the URL string itself. A simple MLP over these
standardized numeric features is the right model here — a char-CNN/BiLSTM
(see url_encoder.py, kept for reference / other datasets) would be solving
a problem this data doesn't have.

Expected feature order (must match FEATURE_COLUMNS in data/url_features.py):
    entropy, url_length, num_www, ratio_digits_url, num_subdomains,
    length_hostname, random_domain, num_dots, num_hyphens,
    ratio_digits_host, prefix_suffix, num_question_mark, num_eq,
    domain_age, dns_resolvable, num_ips, ttl, num_name_servers
"""

import torch
import torch.nn as nn


class URLFeatureEncoder(nn.Module):
    def __init__(self, num_features: int = 18, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, num_features) — already standardized
                      (zero mean, unit variance; see data/url_features.py
                      for the StandardScaler fit on the training split).
        Returns:
            embedding: (batch, embedding_dim)
        """
        return self.net(features)


class URLFeatureClassifierHead(nn.Module):
    """Standalone classifier for training/evaluating the URL branch alone,
    before it's used in the image+URL fusion model."""

    def __init__(self, embedding_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)


if __name__ == "__main__":
    encoder = URLFeatureEncoder(num_features=18)
    head = URLFeatureClassifierHead()
    dummy = torch.randn(4, 18)  # batch of 4, 18 standardized features
    emb = encoder(dummy)
    logits = head(emb)
    print("URL feature embedding shape:", emb.shape)  # (4, 256)
    print("Logits shape:", logits.shape)  # (4, 2)
