"""
fidelity_gate_v2.py — Dual-Gated Reconstruction Head (FidelityGatedHeadV2).

Implements Dual Gating:
  G_structure = Sigmoid(Conv_edge(F))    (Protects genuine microscopic edges)
  G_noise     = Sigmoid(Conv_noise(F))   (Identifies unreliable noise regions)
  G_total     = G_structure * (1.0 - G_noise)

Output:
  Y_restored = Bicubic_base + G_total * Residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FidelityGatedHeadV2(nn.Module):
    """
    Dual-Gated Residual Reconstruction Head:
      Separate edge-protection gate and noise-suppression gate.
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 1, scale: int = 2):
        super().__init__()
        self.scale = scale
        
        # Sub-pixel upsampler for predicted residual
        self.upsample_res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # Structure Protection Gate (High confidence on genuine line edges)
        self.struct_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        nn.init.constant_(self.struct_gate[-1].bias, 2.0)  # Sigmoid(2.0) ~ 0.88 initial weight

        # Noise Suppression Gate (High activation on unreliable noisy regions)
        self.noise_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        nn.init.constant_(self.noise_gate[-1].bias, -2.0)  # Sigmoid(-2.0) ~ 0.12 initial noise dampening

    def forward(self, feat: torch.Tensor, raw_input: torch.Tensor):
        bicubic_base = F.interpolate(raw_input, scale_factor=self.scale, mode="bicubic", align_corners=False)
        pred_residual = self.upsample_res(feat)
        
        g_struct = torch.sigmoid(self.struct_gate(feat))
        g_noise = torch.sigmoid(self.noise_gate(feat))
        
        g_total = g_struct * (1.0 - g_noise)
        
        output = bicubic_base + g_total * pred_residual
        return output, g_total, (g_struct, g_noise)


if __name__ == "__main__":
    head = FidelityGatedHeadV2(in_channels=64, out_channels=1, scale=2)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"FidelityGatedHeadV2 Parameter Count: {n_params:,}")
    
    f_dummy = torch.randn(2, 64, 128, 128)
    x_dummy = torch.randn(2, 1, 128, 128)
    out, g_tot, (g_s, g_n) = head(f_dummy, x_dummy)
    print(f"Output Shape: {out.shape} | Total Gate: {g_tot.shape} | Struct Gate Mean: {g_s.mean():.4f}")
