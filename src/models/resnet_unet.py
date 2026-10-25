"""ResNet18 encoder + U-Net decoder for V12 crop refiner (256px)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

BACKBONES = {
    18: (resnet18, ResNet18_Weights),
    34: (resnet34, ResNet34_Weights),  # same stage widths as 18, deeper stages
}


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


class ResNet18UNet(nn.Module):
    """1-channel input; ResNet18 encoder; binary logits out."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, pretrained: bool = True, depth: int = 18) -> None:
        super().__init__()
        ctor, wenum = BACKBONES[depth]
        weights = wenum.DEFAULT if pretrained else None
        backbone = ctor(weights=weights)

        # Adapt first conv for grayscale
        old = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        elif pretrained and in_channels == 3:
            self.conv1.weight = old.weight

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64
        self.layer2 = backbone.layer2  # 128
        self.layer3 = backbone.layer3  # 256
        self.layer4 = backbone.layer4  # 512

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = ConvBlock(32, 32)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,1,H,W  (expect H=W=256)
        x0 = self.relu(self.bn1(self.conv1(x)))  # /2
        x0p = self.maxpool(x0)  # /4
        e1 = self.layer1(x0p)  # /4, 64
        e2 = self.layer2(e1)  # /8, 128
        e3 = self.layer3(e2)  # /16, 256
        e4 = self.layer4(e3)  # /32, 512

        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)
        # match x0 spatial (/2)
        if d1.shape[-2:] != x0.shape[-2:]:
            d1 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        d0 = self.up0(d1)
        if d0.shape[-2:] != x.shape[-2:]:
            d0 = F.interpolate(d0, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(d0)
        return self.head(d0)
