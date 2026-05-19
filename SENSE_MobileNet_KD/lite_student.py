from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


class MobileNetDensityStudent(nn.Module):
    def __init__(self, variant: str = "small", pretrained: bool = True, feature_dim: int = 128) -> None:
        super().__init__()
        if variant == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            in_channels = 576
        elif variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
            in_channels = 960
        else:
            raise ValueError("variant must be 'small' or 'large'")

        self.variant = variant
        self.features = backbone.features
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        feat = self.feature_proj(x)
        density = torch.relu(self.head(feat))
        return {"density_map": density, "feature_map": feat}
