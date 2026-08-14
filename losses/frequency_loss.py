"""
frequency_loss.py — 2D Real FFT Frequency Domain Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFrequencyLoss(nn.Module):
    """Penalizes high-frequency magnitude spectrum discrepancies in 2D FFT domain."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)

        fft_p = torch.fft.rfft2(p_fp32, norm="ortho")
        fft_t = torch.fft.rfft2(t_fp32, norm="ortho")
        
        mag_p = torch.abs(fft_p)
        mag_t = torch.abs(fft_t)
        return F.l1_loss(mag_p, mag_t)
