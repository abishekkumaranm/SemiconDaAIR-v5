"""
total_loss.py — Modular Composite Loss Function for SemiconDaAIR-v6 (with MoE Load Balancing).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_root_losses_path = os.path.join(_root_dir, "losses.py")

_spec = importlib.util.spec_from_file_location("root_losses", _root_losses_path)
_root_losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_losses)

SSIMLoss = _root_losses.SSIMLoss
MultiScaleSSIMLoss = _root_losses.MultiScaleSSIMLoss
DefectPreservationLoss = _root_losses.DefectPreservationLoss
MultiLabelDegradationLoss = _root_losses.MultiLabelDegradationLoss

from losses.pixel_loss import CharbonnierLoss
from losses.structure_loss import SobelGradientEdgeLoss, LaplacianStructureLoss
from losses.frequency_loss import FourierFrequencyLoss


class SemiconDaAIRv6CompositeLoss(nn.Module):
    """
    Modular Composite Loss Function supporting all ablation configurations EXP-A to EXP-H
    with MoE Expert Load Balancing Loss.
    """
    def __init__(
        self,
        use_charbonnier: bool = True,
        lambda_sobel: float = 0.20,
        lambda_laplacian: float = 0.0,
        lambda_frequency: float = 0.15,
        lambda_ssim: float = 0.10,
        lambda_defect: float = 0.20,
        lambda_moe: float = 0.01
    ):
        super().__init__()
        self.use_charbonnier = use_charbonnier
        self.lambda_sobel = lambda_sobel
        self.lambda_laplacian = lambda_laplacian
        self.lambda_frequency = lambda_frequency
        self.lambda_ssim = lambda_ssim
        self.lambda_defect = lambda_defect
        self.lambda_moe = lambda_moe

        self.charbonnier = CharbonnierLoss()
        self.sobel = SobelGradientEdgeLoss()
        self.laplacian = LaplacianStructureLoss()
        self.fourier = FourierFrequencyLoss()
        self.ms_ssim = MultiScaleSSIMLoss()
        self.defect = DefectPreservationLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, moe_gates: torch.Tensor = None):
        loss_dict = {}

        if self.use_charbonnier:
            l_pixel = self.charbonnier(pred, target)
            loss_dict["charbonnier"] = l_pixel.item()
        else:
            l_pixel = F.l1_loss(pred, target)
            loss_dict["l1"] = l_pixel.item()

        total = l_pixel

        if self.lambda_sobel > 0:
            l_sobel = self.sobel(pred, target)
            total = total + self.lambda_sobel * l_sobel
            loss_dict["sobel"] = l_sobel.item()

        if self.lambda_laplacian > 0:
            l_lap = self.laplacian(pred, target)
            total = total + self.lambda_laplacian * l_lap
            loss_dict["laplacian"] = l_lap.item()

        if self.lambda_frequency > 0:
            l_freq = self.fourier(pred, target)
            total = total + self.lambda_frequency * l_freq
            loss_dict["frequency"] = l_freq.item()

        if self.lambda_ssim > 0:
            l_ssim = self.ms_ssim(pred, target)
            total = total + self.lambda_ssim * l_ssim
            loss_dict["ssim"] = l_ssim.item()

        if self.lambda_defect > 0:
            l_defect = self.defect(pred, target)
            total = total + self.lambda_defect * l_defect
            loss_dict["defect"] = l_defect.item()

        # MoE Load Balancing Loss
        if moe_gates is not None and self.lambda_moe > 0:
            # moe_gates: (b, num_experts)
            num_experts = moe_gates.shape[1]
            mean_gates = moe_gates.mean(dim=0)
            l_moe = num_experts * torch.sum(mean_gates * mean_gates)
            total = total + self.lambda_moe * l_moe
            loss_dict["moe_balance"] = l_moe.item()

        loss_dict["total_loss"] = total.item()
        return total, loss_dict


if __name__ == "__main__":
    criterion = SemiconDaAIRv6CompositeLoss()
    p = torch.randn(2, 1, 128, 128, requires_grad=True)
    t = torch.randn(2, 1, 128, 128)
    gates = torch.softmax(torch.randn(2, 3), dim=-1)
    loss, details = criterion(p, t, moe_gates=gates)
    print("V6 MoE Composite Loss Check:", details)
