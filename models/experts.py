"""
experts.py — Shared and Specialized Low-Rank Experts for SemiconDaAIR.

Includes:
  - LowRankExpertBlock: Base bottleneck block C -> C/r -> C (r=4, 8, 16)
  - SharedExpert: Learns common semiconductor structures (edges, contours, periodic Manhattan line-space patterns)
  - SpeckleExpert: Multiplicative noise suppression & local structure preservation
  - GaussianExpert: Additive noise & defocus blur suppression, contrast restoration
  - ResolutionExpert: High-frequency detail recovery for 2x super-resolution
  - ExpertFusion: Multi-label weighted fusion of shared + specialized expert outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankExpertBlock(nn.Module):
    """
    Low-rank expert projection block: C -> C/r -> C.
    Reduces parameter overhead while allowing expert specialization.
    """
    def __init__(self, channels: int = 64, reduction: int = 8, kernel_size: int = 3):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        padding = kernel_size // 2
        
        self.proj_down = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.dw_conv = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=kernel_size,
            padding=padding, groups=mid_channels
        )
        self.act = nn.SiLU()
        self.proj_up = nn.Conv2d(mid_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj_down(x)
        res = self.dw_conv(res)
        res = self.act(res)
        res = self.proj_up(res)
        return x + res


class SharedExpert(nn.Module):
    """
    Shared/Agnostic Expert.
    Learns common semiconductor structural priors (lines, periodic pitch, boundaries, Manhattan geometry).
    """
    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        self.block1 = LowRankExpertBlock(channels, reduction=reduction, kernel_size=3)
        # 1D coordinate direction pooling for horizontal & vertical line-space arrays
        mid = max(8, channels // reduction)
        self.conv_c = nn.Conv2d(channels, mid, kernel_size=1)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_block = self.block1(x)
        # Coordinate attention spatial guidance
        x_h = x_block.mean(dim=3, keepdim=True)
        x_w = x_block.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        h = x_h.shape[2]
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.conv_c(y))
        y_h = y[:, :, :h, :]
        y_w = y[:, :, h:, :].permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x_block * a_h * a_w


class SpeckleExpert(nn.Module):
    """
    Speckle Noise Expert.
    Specialized for multiplicative laser/e-beam speckle noise suppression without over-smoothing.
    """
    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        self.low_rank1 = LowRankExpertBlock(channels, reduction=reduction, kernel_size=5)
        self.low_rank2 = LowRankExpertBlock(channels, reduction=reduction, kernel_size=3)
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.low_rank1(x)
        feat = self.low_rank2(feat)
        gate = torch.sigmoid(self.gate_conv(feat))
        return feat * gate


class GaussianExpert(nn.Module):
    """
    Gaussian Noise / Defocus Blur Expert.
    Specialized for additive noise & hazy defocus blur suppression, contrast restoration.
    """
    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x[None, None, :, :])
        self.register_buffer("sobel_y", sobel_y[None, None, :, :])
        
        self.low_rank = LowRankExpertBlock(channels, reduction=reduction, kernel_size=3)
        self.edge_gate = nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.low_rank(x)
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.to(x.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(x.dtype), padding=1)
        grad_concat = torch.cat([gx, gy], dim=1)
        
        gate = torch.sigmoid(self.edge_gate(torch.cat([feat, grad_concat], dim=1)))
        return feat * gate


class ResolutionExpert(nn.Module):
    """
    Super-Resolution (SR) Expert.
    Specialized for high-frequency detail recovery and 2x sub-pixel reconstruction.
    """
    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        self.low_rank1 = LowRankExpertBlock(channels, reduction=reduction, kernel_size=3)
        self.conv_hf = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.act = nn.SiLU()
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.low_rank1(x)
        hf = self.act(self.conv_hf(feat))
        return self.proj_out(hf) + feat


class ExpertFusion(nn.Module):
    """
    Multi-Label Expert Fusion Layer.
    Combines Shared Expert and weighted Specialized Experts:
      F_specialized = g_speckle * F_speckle + g_gaussian * F_gaussian + g_sr * F_sr
      F_out = F_shared + F_specialized -> 1x1 Fusion Conv with Residual
    """
    def __init__(self, channels: int = 64, reduction: int = 8):
        super().__init__()
        self.shared_expert = SharedExpert(channels, reduction=reduction)
        self.speckle_expert = SpeckleExpert(channels, reduction=reduction)
        self.gaussian_expert = GaussianExpert(channels, reduction=reduction)
        self.resolution_expert = ResolutionExpert(channels, reduction=reduction)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, C, H, W]
        gates: [B, 3] where [:, 0]=speckle, [:, 1]=gaussian, [:, 2]=resolution
        """
        w_speckle = gates[:, 0].view(-1, 1, 1, 1)
        w_gaussian = gates[:, 1].view(-1, 1, 1, 1)
        w_sr = gates[:, 2].view(-1, 1, 1, 1)
        
        f_shared = self.shared_expert(x)
        f_speckle = self.speckle_expert(x)
        f_gaussian = self.gaussian_expert(x)
        f_sr = self.resolution_expert(x)
        
        f_specialized = w_speckle * f_speckle + w_gaussian * f_gaussian + w_sr * f_sr
        f_out = f_shared + f_specialized
        return x + self.fuse_conv(f_out)


if __name__ == "__main__":
    fusion = ExpertFusion(channels=64, reduction=8)
    dummy_x = torch.randn(2, 64, 32, 32)
    dummy_gates = torch.tensor([[0.8, 0.1, 0.9], [0.2, 0.9, 0.5]])
    out = fusion(dummy_x, dummy_gates)
    print(f"ExpertFusion Test -> Input: {dummy_x.shape} | Gates: {dummy_gates.shape} -> Output: {out.shape}")
