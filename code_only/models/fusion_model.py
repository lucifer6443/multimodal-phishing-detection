"""
Image + URL Fusion Model for Phishing Detection
------------------------------------------------------
Combines screenshot embeddings (ResNet/ViT) and URL-feature embeddings
(MLP over 18 engineered features) via concatenation + FC layers.

NOTE ON SCOPE: the email-text data in this project is NOT row-aligned
with the screenshots or URL features (different collection sources, no
shared ID/domain). So text is trained and evaluated as its own standalone
classifier (see models/text_encoder.py + TextClassifierHead) rather than
fused here. This is a legitimate "late fusion" / multi-classifier design:
your final system runs all three classifiers and combines their
predictions (e.g. majority vote, or a small meta-classifier over the
three probability outputs) rather than one joint model over all three
modalities at once. Report this design choice explicitly in your
methodology — it's a direct, honest consequence of the datasets not
being collected as matched triplets.
"""

import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder
from models.url_feature_encoder import URLFeatureEncoder


class ImageURLFusionDetector(nn.Module):
    def __init__(
        self,
        image_backbone: str = "resnet50",
        num_url_features: int = 18,
        embedding_dim: int = 256,
        num_classes: int = 2,
        freeze_image: bool = False,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone=image_backbone,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_image,
        )
        self.url_encoder = URLFeatureEncoder(
            num_features=num_url_features,
            embedding_dim=embedding_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        image_emb = self.image_encoder(images)
        url_emb = self.url_encoder(url_features)
        fused = torch.cat([image_emb, url_emb], dim=1)
        return self.classifier(fused)


if __name__ == "__main__":
    model = ImageURLFusionDetector()
    images = torch.randn(4, 3, 224, 224)
    url_features = torch.randn(4, 18)  # standardized features
    logits = model(images, url_features)
    print("Logits shape:", logits.shape)  # (4, 2)
