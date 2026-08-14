"""
semicon_daair_v6.py — SemiconDaAIR-v6: Multi-Structural MoE & OOD Generalization Network.

Architecture Pipeline:
  Input LQ (1 channel, signed float32, range exceeding [0,1])
    ↓
  Homomorphic Dynamic Range Handler (Log domain transform: ln(|X| + eps))
    ↓
  Shallow Feature Encoder + Multi-Scale Gated Speckle Branch
    ↓
  Multi-Structural Mixture-of-Experts (Line-Space, Contact-Hole, Interconnect Experts)
    ↓
  Lightweight Structural Guidance Branch (Sobel X/Y + Laplacian + Gradient Magnitude)
    ↓
  Selective Frequency Mining Branch (2D Real FFT Spectral Filtering)
    ↓
  Dual-Stage Bottleneck State-Space Global Context (SSM)
    ↓
  Confidence-Gated PixelShuffle x2 Reconstruction Head (Bounded [0, 1] Output)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.robust_range import HomomorphicRangeHandler
from models.structural_moe import MultiStructuralMoE
from models.structure_guidance import StructuralGuidanceModule
from models.semicon_daair_v3 import SpeckleAwareBranch, FidelityGatedHead, UnlabeledDegradationFingerprint
from models.degradation_router import DegradationRouter
from models.experts import SharedExpert, SpeckleExpert, GaussianExpert, ResolutionExpert
from models.frequency_module import SelectiveFrequencyModule
from models.controller import SelfLearnableController


class DualStageBottleneckSSM(nn.Module):
    """Dual-Stage Bottleneck State-Space Global Context Block for v6 (64 -> 32 -> 64)."""
    def __init__(self, channels: int = 64, reduction: int = 2):
        super().__init__()
        mid_ch = channels // reduction
        self.stage1_in = nn.Conv2d(channels, mid_ch, kernel_size=1)
        self.stage1_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_ch // 2, mid_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.stage2_conv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.stage2_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_ch // 2, mid_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_out = nn.Conv2d(mid_ch, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.stage1_in(x)
        w1 = self.stage1_attn(y)
        y1 = y * w1
        y2 = F.silu(self.stage2_conv(y1))
        w2 = self.stage2_attn(y2)
        out = self.conv_out(y2 * w2)
        return res + out


class SemiconDaAIRv6(nn.Module):
    """
    SemiconDaAIR-v6: Structure-Preserving Lightweight Semiconductor Image Restoration Network.
    Target Parameter Count: ~680K - 750K parameters.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        scale: int = 2
    ):
        super().__init__()
        self.scale = scale
        self.base_channels = base_channels

        # 1. Homomorphic Range Handler & Shallow Feature Encoder
        self.range_handler = HomomorphicRangeHandler(channels=base_channels)
        self.shallow_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.speckle_branch = SpeckleAwareBranch(channels=base_channels)

        # 2. Multi-Structural Mixture-of-Experts (Line-Space, Contact-Hole, Interconnect)
        self.structural_moe = MultiStructuralMoE(channels=base_channels, num_experts=3)

        # 3. Structure & Frequency Mining Branches
        self.spatial_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.structural_guidance = StructuralGuidanceModule(channels=base_channels)
        self.frequency_branch = SelectiveFrequencyModule(channels=base_channels)
        self.fusion_block = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        # 4. Dual-Stage Bottleneck SSM Global Context
        self.global_context = DualStageBottleneckSSM(channels=base_channels, reduction=2)
        self.controller = SelfLearnableController(channels=base_channels, num_heads=4)

        # 5. Confidence-Gated Reconstruction Head
        self.sr_head = FidelityGatedHead(in_channels=base_channels, out_channels=out_channels, scale=scale)

    def forward(self, x: torch.Tensor, return_extras: bool = False):
        norm_feat = self.range_handler(x)
        f0 = self.shallow_conv(x) + norm_feat
        f_speckle = self.speckle_branch(f0, x)

        # Structural MoE routing
        f_moe, moe_gates = self.structural_moe(f_speckle)

        # Structural and frequency refinement
        f_struct = self.structural_guidance(self.spatial_conv(f_moe), x)
        f_freq = self.frequency_branch(f_moe)
        f_fused = self.fusion_block(torch.cat([f_struct, f_freq], dim=1))

        f_out = self.global_context(f_fused)
        f_guided = self.controller(f_fused, f_out)

        out_hr, confidence_map = self.sr_head(f_guided, x)

        if return_extras:
            return out_hr, {
                "confidence_map": confidence_map,
                "moe_gates": moe_gates
            }
        return out_hr


def build_semicon_daair_v6(scale: int = 2) -> SemiconDaAIRv6:
    return SemiconDaAIRv6(scale=scale)


if __name__ == "__main__":
    m = build_semicon_daair_v6(scale=2)
    n_params = sum(p.numel() for p in m.parameters())
    print("=" * 75)
    print("      SemiconDaAIR-v6 Structural-MoE Model Initialized")
    print(f"      Total Parameters : {n_params:,} (Target Budget: 600K-900K)")
    print("=" * 75)
    x = torch.randn(2, 1, 128, 128)
    x[0, 0, 10, 10] = 3.8500
    x[0, 0, 20, 20] = -0.2784
    y = m(x)
    print(f"      Input: {x.shape} -> Output: {y.shape}")
