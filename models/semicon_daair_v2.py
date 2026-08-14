"""
semicon_daair_v2.py — SemiconDaAIR-v2 Architecture for Semiconductor Image Restoration.

Conceptual Pipeline:
  128x128 Float32 Input
        |
  Shallow Feature Encoder + Speckle-Aware Branch (Signed Log Transform)
        |
  Continuous Degradation Representation (FiLM Modulation: gamma(d)*F + beta(d))
        |
  Multi-label Router (Sigmoid Gating)
        |
  [ Gaussian Expert | Speckle Expert | SR Expert ]
        |
  Shared Expert
        |
  Spatial Branch (Edge/Gradient Guidance) + Frequency Branch (2D FFT)
        |
  Fusion Block & Self-Learnable Controller
        |
  PixelShuffle 2x Super-Resolution & Global Residual Head
        |
  256x256 Float32 Restored Output
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .degradation_router import DegradationRouter
from .experts import SharedExpert, SpeckleExpert, GaussianExpert, ResolutionExpert
from .frequency_module import SelectiveFrequencyModule
from .edge_module import EdgeGuidanceModule
from .controller import SelfLearnableController
from .sr_head import PixelShuffleSRHead


class SpeckleAwareBranch(nn.Module):
    """
    Speckle-Aware Branch using safe signed log transform:
      signed_log(x) = sign(x) * log(1 + abs(x))
    Converts multiplicative speckle noise Y = X * (1 + N) into additive space.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv_log = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1)
        )
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, spatial_feat: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        signed_log_x = torch.sign(raw_input) * torch.log1p(torch.abs(raw_input))
        log_feat = self.conv_log(signed_log_x)
        gate = self.fuse_gate(torch.cat([spatial_feat, log_feat], dim=1))
        return spatial_feat + gate * log_feat


class ContinuousDegradationEncoder(nn.Module):
    """
    Estimates continuous degradation representation embedding d and generates FiLM modulation parameters (gamma, beta).
    """
    def __init__(self, in_channels: int = 64, embed_dim: int = 64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp_deg = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.film_gen = nn.Linear(embed_dim, in_channels * 2)

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        z = self.gap(x).view(b, c)
        deg_emb = self.mlp_deg(z)
        
        gamma_beta = self.film_gen(deg_emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        modulated_x = x * (1.0 + gamma) + beta
        return modulated_x, deg_emb


class SemiconDaAIRv2(nn.Module):
    """
    SemiconDaAIR-v2: Advanced Physics-Guided Degradation-Adapted Restoration Network.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 4,
        scale: int = 2,
        low_rank_reduction: int = 8
    ):
        super().__init__()
        self.scale = scale
        self.base_channels = base_channels

        # 1. Shallow Feature Encoder & Speckle-Aware Branch
        self.shallow_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.speckle_branch = SpeckleAwareBranch(channels=base_channels)

        # 2. Continuous Degradation Encoder & Multi-label Router
        self.deg_encoder = ContinuousDegradationEncoder(in_channels=base_channels, embed_dim=64)
        self.router = DegradationRouter(in_channels=base_channels, hidden_dim=32, num_degradations=3)

        # 3. Specialized & Shared Low-Rank Experts
        self.gaussian_expert = GaussianExpert(base_channels, reduction=low_rank_reduction)
        self.speckle_expert = SpeckleExpert(base_channels, reduction=low_rank_reduction)
        self.sr_expert = ResolutionExpert(base_channels, reduction=low_rank_reduction)
        self.shared_expert = SharedExpert(base_channels, reduction=low_rank_reduction)

        # 4. Spatial & Frequency Mining Branches
        self.spatial_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.edge_guidance = EdgeGuidanceModule(channels=base_channels)
        self.frequency_branch = SelectiveFrequencyModule(channels=base_channels)
        self.fusion_block = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        # 5. Bottleneck & Self-Learnable Controller
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        )
        self.controller = SelfLearnableController(channels=base_channels, num_heads=4)

        # 6. PixelShuffle 2x Super-Resolution & Global Residual Head
        self.sr_head = PixelShuffleSRHead(in_channels=base_channels, out_channels=out_channels, upscale=scale)

    def forward(self, x: torch.Tensor, return_router_logits: bool = False):
        """
        Input x: [B, 1, H, W]
        Returns:
          out_hr: [B, 1, scale*H, scale*W]
          (Optional) logits: [B, 3] for multi-label degradation loss
        """
        # Shallow features & Speckle-Aware Transformation
        f0 = self.shallow_conv(x)
        f_speckle_aware = self.speckle_branch(f0, x)

        # Continuous Degradation FiLM Modulation & Router
        f_film, deg_emb = self.deg_encoder(f_speckle_aware)
        gates, logits = self.router(f_film)

        # Expert Execution & Fusion
        w_gaussian = gates[:, 0].view(-1, 1, 1, 1)
        w_speckle = gates[:, 1].view(-1, 1, 1, 1)
        w_sr = gates[:, 2].view(-1, 1, 1, 1)

        f_g = self.gaussian_expert(f_film)
        f_s = self.speckle_expert(f_film)
        f_sr = self.sr_expert(f_film)

        f_specialized = w_gaussian * f_g + w_speckle * f_s + w_sr * f_sr
        f_shared = self.shared_expert(f_film + f_specialized)

        # Dual Spatial & Frequency Branches
        f_spatial = self.edge_guidance(self.spatial_conv(f_shared), x)
        f_freq = self.frequency_branch(f_shared)
        f_fused = self.fusion_block(torch.cat([f_spatial, f_freq], dim=1))

        # Bottleneck & Controller
        f_bottle = self.bottleneck(f_fused)
        f_guided = self.controller(f_fused, f_bottle)

        # Super-Resolution Head & Global Residual Reconstruction
        out_hr = self.sr_head(f_guided, x)

        if return_router_logits or (self.training and return_router_logits):
            return out_hr, logits
        return out_hr

    def forward_with_details(self, x: torch.Tensor):
        """Forward pass returning restored output and actual router gate weights."""
        f0 = self.shallow_conv(x)
        f_speckle_aware = self.speckle_branch(f0, x)

        f_film, deg_emb = self.deg_encoder(f_speckle_aware)
        gates, logits = self.router(f_film)

        w_gaussian = gates[:, 0].view(-1, 1, 1, 1)
        w_speckle = gates[:, 1].view(-1, 1, 1, 1)
        w_sr = gates[:, 2].view(-1, 1, 1, 1)

        f_g = self.gaussian_expert(f_film)
        f_s = self.speckle_expert(f_film)
        f_sr = self.sr_expert(f_film)

        f_specialized = w_gaussian * f_g + w_speckle * f_s + w_sr * f_sr
        f_shared = self.shared_expert(f_film + f_specialized)

        f_spatial = self.edge_guidance(self.spatial_conv(f_shared), x)
        f_freq = self.frequency_branch(f_shared)
        f_fused = self.fusion_block(torch.cat([f_spatial, f_freq], dim=1))

        f_bottle = self.bottleneck(f_fused)
        f_guided = self.controller(f_fused, f_bottle)

        out_hr = self.sr_head(f_guided, x)

        router_dict = {
            "gaussian": float(gates[0, 0].item()),
            "speckle": float(gates[0, 1].item()),
            "sr": float(gates[0, 2].item())
        }
        return out_hr, router_dict

    @torch.no_grad()
    def forward_self_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Geometric self-ensemble (x8 TTA) inference."""
        def _transform(t, op):
            if op & 1:
                t = torch.flip(t, dims=[-1])
            if op & 2:
                t = torch.flip(t, dims=[-2])
            if op & 4:
                t = torch.transpose(t, -2, -1)
            return t

        def _inverse(t, op):
            if op & 4:
                t = torch.transpose(t, -2, -1)
            if op & 2:
                t = torch.flip(t, dims=[-2])
            if op & 1:
                t = torch.flip(t, dims=[-1])
            return t

        out_list = []
        for op in range(8):
            x_aug = _transform(x, op)
            out_aug = self.forward(x_aug)
            out_list.append(_inverse(out_aug, op))

        return torch.stack(out_list, dim=0).mean(dim=0)


def build_semicon_daair_v2(scale=2, base_channels=64, num_blocks=4, low_rank_reduction=8):
    return SemiconDaAIRv2(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        num_blocks=num_blocks,
        scale=scale,
        low_rank_reduction=low_rank_reduction
    )


if __name__ == "__main__":
    m = build_semicon_daair_v2(scale=2, base_channels=64)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"SemiconDaAIR-v2 Parameter Count: {n_params:,}")
    dummy_x = torch.randn(2, 1, 128, 128)
    out_hr, logits = m(dummy_x, return_router_logits=True)
    print(f"Forward Test -> Input: {dummy_x.shape} -> Output HR: {out_hr.shape} | Logits: {logits.shape}")
