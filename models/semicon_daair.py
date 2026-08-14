"""
semicon_daair.py — SemiconDaAIR: Physics-Guided Degradation-Adapted Restoration Network for Semiconductor Inspection.

Adapted from:
  "Efficient Degradation-aware Any Image Restoration" (Zamfir et al., arXiv:2405.15475)

Key Architectural Features:
  1. Shallow Feature Extraction: Convolutions mapping raw grayscale float32 input to base embedding dimension.
  2. Spatial & Frequency Sub-band Mining: 2D FFT sub-band analysis (SelectiveFrequencyModule) + Spatial Convolutions.
  3. Degradation Encoder: Global average pooling + pretrained DnCNN residual prior.
  4. Multi-Label Degradation Router: Sigmoid gating for simultaneous Speckle AND Gaussian AND SR degradations.
  5. Low-Rank Shared & Specialized Experts: Shared, Speckle, Gaussian, and Resolution Experts (ExpertFusion).
  6. Edge-Preservation Guidance Path: Sobel/Laplacian edge guidance without artificial structure hallucination.
  7. Self-Learnable Controller: Cross-attention query-key-value guidance from bottleneck to decoder.
  8. PixelShuffle SR Head & Global Residual Reconstruction: 2x sub-pixel convolution upsampling over bicubic baseline.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .degradation_router import DegradationRouter
from .experts import ExpertFusion
from .frequency_module import SelectiveFrequencyModule
from .edge_module import EdgeGuidanceModule
from .controller import SelfLearnableController
from .sr_head import PixelShuffleSRHead


class SemiconDaAIR(nn.Module):
    """
    SemiconDaAIR: Degradation-Adapted Multi-Label Semiconductor Restoration Network.
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
        
        # 1. Shallow Feature Extraction
        self.shallow_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 2. Spatial & Frequency Mining Branches
        self.spatial_branch = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.frequency_branch = SelectiveFrequencyModule(channels=base_channels)
        self.fuse_branches = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        # 3. Degradation Encoder & Multi-label Router
        self.router = DegradationRouter(in_channels=base_channels, hidden_dim=32, num_degradations=3)

        # 4. Shared + Specialized Low-Rank Experts
        self.encoder_blocks = nn.ModuleList([
            ExpertFusion(channels=base_channels, reduction=low_rank_reduction)
            for _ in range(num_blocks)
        ])

        # 5. Bottleneck & Controller
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        )
        self.controller = SelfLearnableController(channels=base_channels, num_heads=4)

        # 6. Decoder Blocks
        self.decoder_blocks = nn.ModuleList([
            ExpertFusion(channels=base_channels, reduction=low_rank_reduction)
            for _ in range(num_blocks)
        ])

        # 7. Edge Preservation Guidance Path
        self.edge_guidance = EdgeGuidanceModule(channels=base_channels)

        # 8. Super-Resolution Head & Global Residual Reconstruction
        self.sr_head = PixelShuffleSRHead(in_channels=base_channels, out_channels=out_channels, upscale=scale)

    def forward(self, x: torch.Tensor, return_router_logits: bool = False):
        """
        Input x: [B, 1, H, W]
        Returns:
          out_hr: [B, 1, scale*H, scale*W]
          (Optional) logits: [B, 3] for multi-label degradation loss
        """
        # Shallow features
        f0 = self.shallow_conv(x)
        
        # Spatial & Frequency Dual Branches
        f_spatial = self.spatial_branch(f0)
        f_freq = self.frequency_branch(f0)
        f_fused = self.fuse_branches(torch.cat([f_spatial, f_freq], dim=1))
        
        # Multi-Label Degradation Router
        gates, logits = self.router(f_fused)
        
        # Encoder Pass through Expert Fusion
        f_enc = f_fused
        for block in self.encoder_blocks:
            f_enc = block(f_enc, gates)
            
        # Bottleneck & Self-Learnable Controller
        f_bottle = self.bottleneck(f_enc)
        
        # Decoder Pass guided by Controller Cross-Attention
        f_dec = f_bottle
        for block in self.decoder_blocks:
            f_dec = block(f_dec, gates)
        f_guided = self.controller(f_dec, f_bottle)
        
        # High-Frequency Edge Guidance Path
        f_edge = self.edge_guidance(f_guided, x)
        
        # 2x Super-Resolution & Global Residual Reconstruction
        out_hr = self.sr_head(f_edge, x)
        
        if return_router_logits or (self.training and return_router_logits):
            return out_hr, logits
        return out_hr

    @torch.no_grad()
    def forward_self_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Geometric self-ensemble (x8 TTA) inference for PSNR/SSIM boosting."""
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


def build_semicon_daair(scale=2, base_channels=64, num_blocks=4, low_rank_reduction=8):
    return SemiconDaAIR(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        num_blocks=num_blocks,
        scale=scale,
        low_rank_reduction=low_rank_reduction
    )


if __name__ == "__main__":
    model = build_semicon_daair(scale=2, base_channels=64, num_blocks=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SemiconDaAIR Parameter Count: {n_params:,}")
    
    sample_128 = torch.randn(2, 1, 128, 128)
    out_256, logits = model(sample_128, return_router_logits=True)
    print(f"128x128 -> 256x256 Pass: Input {sample_128.shape} -> Output {out_256.shape} | Logits: {logits.shape}")

    sample_256 = torch.randn(1, 1, 256, 256)
    out_512 = model(sample_256)
    print(f"256x256 -> 512x512 Pass: Input {sample_256.shape} -> Output {out_512.shape}")
