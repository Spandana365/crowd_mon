from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class FeatureAligner(nn.Module):
    def __init__(self, student_channels: int) -> None:
        super().__init__()
        self.student_channels = student_channels
        self.teacher_adapter = nn.LazyConv2d(student_channels, kernel_size=1, bias=False)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        teacher_feat = F.interpolate(
            teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        if teacher_feat.shape[1] != self.student_channels:
            teacher_feat = self.teacher_adapter(teacher_feat)
        return F.mse_loss(student_feat, teacher_feat)


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = num / (den + 1e-6)
    return torch.clamp((1.0 - ssim) * 0.5, min=0.0, max=1.0).mean()


def compute_losses(
    student_density: torch.Tensor,
    gt_density: torch.Tensor,
    teacher_density: torch.Tensor,
    l_feat: torch.Tensor,
    l_count: torch.Tensor,
    use_ssim: bool = False,
    ssim_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    mse_gt = F.mse_loss(student_density, gt_density)
    mae_gt = F.l1_loss(student_density, gt_density)
    l_gt = mse_gt + (0.5 * mae_gt)
    if use_ssim and ssim_weight > 0.0:
        l_gt = l_gt + (ssim_weight * _ssim_loss(student_density, gt_density))

    l_teacher = F.mse_loss(student_density, teacher_density)
    total = l_gt + (0.3 * l_teacher) + (0.1 * l_feat) + (0.1 * l_count)
    return {"l_gt": l_gt, "l_teacher": l_teacher, "l_feat": l_feat, "l_count": l_count, "total": total}
