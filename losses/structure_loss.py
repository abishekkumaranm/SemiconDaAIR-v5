"""
structure_loss.py — Structure-Aware Sobel Gradient & Laplacian Loss Suite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradientEdgeLoss(nn.Module):
    """Sobel-gradient edge loss to enforce sharp pattern transition boundaries."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x[None, None, :, :])
        self.register_buffer("sobel_y", sobel_y[None, None, :, :])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)
        sobel_x_buf = self.sobel_x.to(torch.float32)
        sobel_y_buf = self.sobel_y.to(torch.float32)

        gx_p = F.conv2d(p_fp32, sobel_x_buf, padding=1)
        gy_p = F.conv2d(p_fp32, sobel_y_buf, padding=1)
        gx_t = F.conv2d(t_fp32, sobel_x_buf, padding=1)
        gy_t = F.conv2d(t_fp32, sobel_y_buf, padding=1)
        
        mag_p = torch.sqrt(gx_p**2 + gy_p**2 + 1e-6)
        mag_t = torch.sqrt(gx_t**2 + gy_t**2 + 1e-6)
        return F.l1_loss(mag_p, mag_t)


class LaplacianStructureLoss(nn.Module):
    """Laplacian high-pass response loss to preserve high-frequency line spacing."""
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("lap", lap[None, None, :, :])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)
        lap_buf = self.lap.to(torch.float32)

        lap_p = F.conv2d(p_fp32, lap_buf, padding=1)
        lap_t = F.conv2d(t_fp32, lap_buf, padding=1)
        return F.l1_loss(lap_p, lap_t)
