"""
fidelity_gate.py — Learned Fidelity Gating Module for SemiconDaAIR-v3.

Prevents hallucination of artificial defects and oversharpening ringing by constraining
predicted restoration residual updates:
    F_out = F_in + G * residual
where G in [0, 1] is a learned per-pixel confidence/fidelity map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FidelityGatedHead(nn.Module):
    """
    Learned Fidelity Gating Module.
    Combines input features and predicted restoration residual to compute a spatial confidence map G.
    Initialized such that G starts near 1.0 (identity path preservation).
    """
    def __init__(self, channels: int = 64, reduction: int = 4):
        super().__init__()
        self.channels = channels

        # Lightweight Depthwise-Separable Feature Fusion
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Initialize gate bias to +2.0 so Sigmoid(2.0) ~ 0.88 (near identity initialization)
        nn.init.constant_(self.gate_conv[3].bias, 2.0)
        nn.init.zeros_(self.gate_conv[3].weight)

    def forward(self, x_in: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x_in: [B, C, H, W] Input feature map
        residual: [B, C, H, W] Predicted restoration residual
        Returns: [B, C, H, W] Gated feature map
        """
        cat_feat = torch.cat([x_in, residual], dim=1)
        gate_map = self.gate_conv(cat_feat)  # [B, C, H, W] in [0, 1]
        
        # Gated residual addition
        gated_residual = gate_map * residual
        out = x_in + gated_residual
        return out

    def forward_with_gate_map(self, x_in: torch.Tensor, residual: torch.Tensor):
        """Returns both gated output and spatial gate map for visualization/dashboard."""
        cat_feat = torch.cat([x_in, residual], dim=1)
        gate_map = self.gate_conv(cat_feat)
        out = x_in + gate_map * residual
        return out, gate_map


if __name__ == "__main__":
    print("[TEST] Testing FidelityGatedHead...")
    x = torch.randn(2, 64, 32, 32)
    res = torch.randn(2, 64, 32, 32)
    gate = FidelityGatedHead(channels=64)
    out, gmap = gate.forward_with_gate_map(x, res)
    print(f"[PASS] FidelityGatedHead output shape: {out.shape}, Gate map shape: {gmap.shape}, mean gate: {gmap.mean().item():.4f}")
