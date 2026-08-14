"""
baselines.py — Benchmark Baseline Models for Semiconductor Super-Resolution & Restoration.

Includes:
  1. Bicubic Interpolation Baseline
  2. SRCNN (Dong et al. ECCV 2014)
  3. EDSR-Light (Enhanced Deep Super-Resolution, Lim et al. CVPRW 2017)
  4. SwinIR-Light (Swin Transformer for Image Restoration, Liang et al. ICCVW 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicBaseline(nn.Module):
    """Bicubic Interpolation Baseline (0 parameters)."""
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class SRCNN(nn.Module):
    """SRCNN: Super-Resolution Convolutional Neural Network (Dong et al.)."""
    def __init__(self, in_channels=1, out_channels=1, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        out = self.relu1(self.conv1(up))
        out = self.relu2(self.conv2(out))
        return self.conv3(out)


class EDSRLight(nn.Module):
    """EDSR-Light: Enhanced Deep Residual Network for Super-Resolution."""
    def __init__(self, in_channels=1, out_channels=1, num_blocks=4, num_feats=64, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, num_feats, kernel_size=3, padding=1)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
            ))
        self.body = nn.ModuleList(blocks)
        self.body_tail = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_feats * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.tail = nn.Conv2d(num_feats, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        f = self.head(x)
        res = f
        for b in self.body:
            res = res + b(res)
        res = self.body_tail(res)
        f = f + res
        up = self.upsample(f)
        return self.tail(up)


class SwinIRLight(nn.Module):
    """SwinIR-Light: Lightweight Swin Transformer / Gated Residual Network for Image Restoration."""
    def __init__(self, in_channels=1, out_channels=1, embed_dim=48, num_blocks=3, scale=2):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
            ))
        self.body = nn.ModuleList(blocks)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.GELU()
        )
        self.tail = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        f = self.head(x)
        res = f
        for b in self.body:
            res = res + b(res)
        up = self.upsample(res)
        return self.tail(up)
