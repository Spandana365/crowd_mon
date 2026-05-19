from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MobileNetV3DensityStudent(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.proj = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        feature_map = self.proj(x)
        density = self.head(feature_map)
        return {"feature_map": feature_map, "density_map": density}


def feature_distill_loss(student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
    teacher_aligned = F.interpolate(
        teacher_feature,
        size=student_feature.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    if teacher_aligned.shape[1] != student_feature.shape[1]:
        min_channels = min(teacher_aligned.shape[1], student_feature.shape[1])
        teacher_aligned = teacher_aligned[:, :min_channels]
        student_feature = student_feature[:, :min_channels]
    return F.mse_loss(student_feature, teacher_aligned)
