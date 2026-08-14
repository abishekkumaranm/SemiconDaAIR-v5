"""
edge_module.py — Edge Guidance Module for SemiconDaAIR.

Key Features:
  - Directional Sobel / Laplacian edge gradient extraction.
  - Generates structural edge guidance maps to preserve line-end corners, contacts, and LER bounds.
  - Multiplicative feature guidance (F * (1 + SoftAttn)) ensuring edge guidance without artificial edge injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGuidanceModule(nn.Module):
    """
    Lightweight Edge Guidance Module.
    Extracts Sobel gradient magnitude and applies soft multiplicative feature guidance.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = sobel_x.t().contiguous()
        sobel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)  # (2, 1, 3, 3)
        self.register_buffer("sobel", sobel)
        
        self.edge_head = nn.Sequential(
            nn.Conv2d(2, channels // 4 or 1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4 or 1, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, C, H, W]
        raw_input: [B, 1, H, W]
        """
        sobel_buf = self.sobel.to(x.dtype)
        # Compute gradient vectors Gx, Gy
        grad = F.conv2d(raw_input, sobel_buf, padding=1)
        if grad.shape[2:] != x.shape[2:]:
            grad = F.interpolate(grad, size=x.shape[2:], mode="bilinear", align_corners=False)
            
        attn = self.edge_head(grad)
        return x * (1.0 + attn)


if __name__ == "__main__":
    edge_mod = EdgeGuidanceModule(channels=64)
    dummy_x = torch.randn(2, 64, 32, 32)
    dummy_raw = torch.randn(2, 1, 32, 32)
    out = edge_mod(dummy_x, dummy_raw)
    print(f"EdgeGuidanceModule Test -> Features: {dummy_x.shape} | Raw Input: {dummy_raw.shape} -> Output: {out.shape}")
