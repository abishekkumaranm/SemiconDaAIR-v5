"""
pixel_loss.py — Pixel Intensity Loss Suite (L1 and Charbonnier Smooth Loss).
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 smooth variant): sqrt((pred - target)^2 + eps^2)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()
