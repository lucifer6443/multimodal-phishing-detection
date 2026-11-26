"""
Image Encoder for Multimodal Spam/Phishing Detection
-------------------------------------------------------
Encodes webpage screenshots (e.g. from Phish360, 1280x1080) into a fixed-size
embedding using a pretrained CNN (ResNet50) or Vision Transformer (ViT).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet50", embedding_dim: int = 256, freeze_backbone: bool = False):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet50":
            base = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "vit_b_16":
            base = tv_models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            feature_dim = base.heads.head.in_features
            base.heads = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projection(features)


def get_recommended_transform(backbone: str = "resnet50"):
    if backbone == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2.transforms()
    elif backbone == "vit_b_16":
        return ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    raise ValueError(f"Unsupported backbone: {backbone}")


class ImageClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)