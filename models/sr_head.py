"""
sr_head.py — PixelShuffle 2x Super-Resolution & Global Residual Reconstruction Head for SemiconDaAIR.

Key Features:
  - 2x Sub-pixel Convolution Upsampling via PixelShuffle.
  - Global Residual Reconstruction:
      Output = BicubicUpsample(Input) + PredictedResidual
  - Reconstructs missing high-frequency sub-nanometer details without checkerboard artifacts.
  - Outputs exact target ground-truth resolution (128x128 -> 256x256 or 256x256 -> 512x512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleSRHead(nn.Module):
    """
    Sub-Pixel Convolution 2x SR Reconstruction Head with Global Residual Learning.
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 1, upscale: int = 2):
        super().__init__()
        self.upscale = upscale
        
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale),
            nn.PReLU(in_channels)
        )
        self.conv_residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(in_channels // 2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, feat: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, C, H, W]
        raw_input: [B, 1, H, W]
        Returns: [B, 1, upscale*H, upscale*W]
        """
        up_feat = self.upsample(feat)
        predicted_residual = self.conv_residual(up_feat)
        
        # Bicubic baseline upsampling skip
        baseline = F.interpolate(raw_input, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        return baseline + predicted_residual


if __name__ == "__main__":
    sr_head = PixelShuffleSRHead(in_channels=64, out_channels=1, upscale=2)
    dummy_feat = torch.randn(2, 64, 64, 64)
    dummy_input = torch.randn(2, 1, 64, 64)
    output = sr_head(dummy_feat, dummy_input)
    print(f"PixelShuffleSRHead Test -> Input: {dummy_input.shape} | Feat: {dummy_feat.shape} -> HR Output: {output.shape}")
