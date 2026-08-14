"""
degradation_router.py — Multi-Label Degradation-Aware Router for SemiconDaAIR.

Key Features:
  - Extracts global degradation representation via Global Average Pooling (GAP).
  - Lightweight 2-layer MLP predicting multi-label degradation logits for:
      1. Speckle Noise
      2. Gaussian Blur / Noise
      3. Spatial Resolution Reduction (SR)
  - Uses Sigmoid activation so multiple experts can activate simultaneously (Multi-label AND logic).
  - Returns both sigmoid gates (for expert weighting) and raw logits (for BCEWithLogitsLoss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationRouter(nn.Module):
    """
    Multi-label Degradation Router.
    Output gates shape: [B, 3] in range [0, 1]
    Logits shape: [B, 3]
    """
    def __init__(self, in_channels: int = 64, hidden_dim: int = 32, num_degradations: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_degradations)
        )

    def forward(self, x: torch.Tensor):
        """
        Input x: [B, C, H, W]
        Returns:
          gates: [B, 3] (Sigmoid probabilities)
          logits: [B, 3] (Unscaled logits for BCE loss)
        """
        b, c, _, _ = x.shape
        z = self.gap(x).view(b, c)
        logits = self.mlp(z)
        gates = torch.sigmoid(logits)
        return gates, logits


if __name__ == "__main__":
    router = DegradationRouter(in_channels=64)
    dummy_feat = torch.randn(4, 64, 32, 32)
    gates, logits = router(dummy_feat)
    print(f"Router Test -> Dummy Feature: {dummy_feat.shape}")
    print(f"Gates Shape: {gates.shape} | Sample Gates:\n{gates}")
    print(f"Logits Shape: {logits.shape}")
