from typing import Dict, List, Tuple

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetStudent(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoders = nn.ModuleList(
            [
                DoubleConv(in_channels, widths[0]),
                DoubleConv(widths[0], widths[1]),
                DoubleConv(widths[1], widths[2]),
                DoubleConv(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(widths[3], widths[3] * 2)
        bottleneck_channels = widths[3] * 2

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(bottleneck_channels, widths[3], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2),
                nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DoubleConv(widths[3] * 2, widths[3]),
                DoubleConv(widths[2] * 2, widths[2]),
                DoubleConv(widths[1] * 2, widths[1]),
                DoubleConv(widths[0] * 2, widths[0]),
            ]
        )
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.out_activation = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleDict({"encoders": self.encoders, "pool": self.pool})
        self.decoder = nn.ModuleDict(
            {"upconvs": self.upconvs, "decoders": self.decoders, "head": self.head}
        )
        self.bottleneck_channels = bottleneck_channels

    @staticmethod
    def _match_spatial(src: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if src.shape[-2:] == target_hw:
            return src
        return nn.functional.interpolate(src, size=target_hw, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips: List[torch.Tensor] = []
        out = x
        for idx, encoder in enumerate(self.encoders):
            out = encoder(out)
            skips.append(out)
            if idx < len(self.encoders) - 1:
                out = self.pool(out)

        bottleneck = self.bottleneck(self.pool(skips[-1]))
        out = bottleneck

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = up(out)
            out = self._match_spatial(out, skip.shape[-2:])
            out = dec(torch.cat([skip, out], dim=1))

        density_map = self.out_activation(self.head(out))
        return {"density_map": density_map, "bottleneck": bottleneck}
