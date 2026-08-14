"""
degradation_encoder.py — Unsupervised Degradation-Aware Conditioning Encoder.

Estimates noise severity, spatial downsampling characteristics, and dynamic range skewness
directly from input features using Global Average Pooling (GAP) + Standard Deviation Pooling (StdPool).

Applies FiLM affine conditioning:
  F_modulated = gamma(d) * F + beta(d)

No hardcoded labels, no explicit routing tables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnsupervisedDegradationEncoder(nn.Module):
    """
    Unsupervised Degradation Encoder:
      Extracts d in R^embed_dim via GAP + StdPool statistics.
      Outputs FiLM scale (gamma) and shift (beta) parameters.
    """
    def __init__(self, in_channels: int = 64, embed_dim: int = 16):
        super().__init__()
        # Input to MLP is GAP (C) + StdPool (C) = 2*C
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )
        self.film_gen = nn.Linear(embed_dim, in_channels * 2)

        # Initialize gamma to 0, beta to 0 (FiLM identity at initialization: 1*F + 0)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        std = torch.std(x, dim=(-2, -1), keepdim=False)
        z = torch.cat([gap, std], dim=1)

        d = self.mlp(z)
        gamma_beta = self.film_gen(d)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Identity-preserving FiLM conditioning
        modulated_x = x * (1.0 + gamma) + beta
        return modulated_x, d


if __name__ == "__main__":
    enc = UnsupervisedDegradationEncoder(in_channels=64, embed_dim=16)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"UnsupervisedDegradationEncoder Parameter Count: {n_params:,}")

    x_dummy = torch.randn(2, 64, 128, 128)
    out, d = enc(x_dummy)
    print(f"Modulated Output Shape: {out.shape} | Fingerprint Shape: {d.shape}")
