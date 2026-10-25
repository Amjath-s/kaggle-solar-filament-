"""Small U-Net for binary segmentation (fits constrained VRAM)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_channels: int = 1,
        dual_head: bool = False,
    ) -> None:
        super().__init__()
        self.dual_head = bool(dual_head)
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = ConvBlock(c5, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c2, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1) if self.dual_head else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        main = self.head(d1)
        if self.dual_head and self.boundary_head is not None:
            return main, self.boundary_head(d1)
        return main


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    name = name.lower()
    if name == "unet":
        kwargs = {k: v for k, v in kwargs.items() if k in ("in_channels", "base_channels", "out_channels")}
        return UNet(**kwargs)
    if name in ("resnet18_unet", "resnet18-unet", "resnet34_unet", "resnet34-unet"):
        from src.models.resnet_unet import ResNet18UNet

        return ResNet18UNet(
            in_channels=int(kwargs.get("in_channels", 1)),
            out_channels=int(kwargs.get("num_classes", kwargs.get("out_channels", 1))),
            pretrained=bool(kwargs.get("pretrained", True)),
            depth=34 if name.startswith("resnet34") else 18,
        )
    raise ValueError(f"Unknown model: {name}")
