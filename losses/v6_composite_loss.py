"""
v6_composite_loss.py — Modular Loss Function Suite & Controlled Loss Ablation Engine for SemiconDaAIR-v6.

Supported Loss Configurations:
  - Configuration A: L1 Loss
  - Configuration B: L1 + Sobel Edge Loss
  - Configuration C: Charbonnier + Sobel Edge Loss
  - Configuration D: Charbonnier + Sobel Edge + 2D FFT Frequency Loss
  - Configuration E: Charbonnier + Sobel Edge + 2D FFT Frequency + SSIM Loss
  - Configuration F: Full Composite Loss (Charbonnier + Sobel + 2D FFT + MS-SSIM + Defect Preservation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import importlib.util

_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_root_losses_path = os.path.join(_root_dir, "losses.py")

_spec = importlib.util.spec_from_file_location("root_losses", _root_losses_path)
_root_losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_losses)

CharbonnierLoss = _root_losses.CharbonnierLoss
SSIMLoss = _root_losses.SSIMLoss
MultiScaleSSIMLoss = _root_losses.MultiScaleSSIMLoss
SobelGradientEdgeLoss = _root_losses.SobelGradientEdgeLoss
FourierFrequencyLoss = _root_losses.FourierFrequencyLoss
DefectPreservationLoss = _root_losses.DefectPreservationLoss
MultiLabelDegradationLoss = _root_losses.MultiLabelDegradationLoss


class V6AblationLoss(nn.Module):
    """
    Modular Loss Function supporting Ablations A, B, C, D, E, F.
    """
    def __init__(self, config_name: str = "F"):
        super().__init__()
        self.config_name = config_name.upper()

        self.charbonnier = CharbonnierLoss()
        self.sobel = SobelGradientEdgeLoss()
        self.fourier = FourierFrequencyLoss()
        self.ssim = SSIMLoss()
        self.ms_ssim = MultiScaleSSIMLoss()
        self.defect = DefectPreservationLoss()
        self.deg_loss = MultiLabelDegradationLoss(lambda_deg=0.05)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, router_logits=None, target_deg_vector=None):
        loss_dict = {}

        if self.config_name == "A":
            # Configuration A: Pure L1
            total = F.l1_loss(pred, target)
            loss_dict["l1"] = total.item()

        elif self.config_name == "B":
            # Configuration B: L1 + Sobel
            l_l1 = F.l1_loss(pred, target)
            l_sobel = self.sobel(pred, target)
            total = l_l1 + 0.2 * l_sobel
            loss_dict["l1"] = l_l1.item()
            loss_dict["sobel"] = l_sobel.item()

        elif self.config_name == "C":
            # Configuration C: Charbonnier + Sobel
            l_charb = self.charbonnier(pred, target)
            l_sobel = self.sobel(pred, target)
            total = l_charb + 0.2 * l_sobel
            loss_dict["charbonnier"] = l_charb.item()
            loss_dict["sobel"] = l_sobel.item()

        elif self.config_name == "D":
            # Configuration D: Charbonnier + Sobel + Frequency (FFT)
            l_charb = self.charbonnier(pred, target)
            l_sobel = self.sobel(pred, target)
            l_freq = self.fourier(pred, target)
            total = l_charb + 0.2 * l_sobel + 0.15 * l_freq
            loss_dict["charbonnier"] = l_charb.item()
            loss_dict["sobel"] = l_sobel.item()
            loss_dict["fourier"] = l_freq.item()

        elif self.config_name == "E":
            # Configuration E: Charbonnier + Sobel + Frequency + SSIM
            l_charb = self.charbonnier(pred, target)
            l_sobel = self.sobel(pred, target)
            l_freq = self.fourier(pred, target)
            l_ssim = self.ssim(pred, target)
            total = l_charb + 0.2 * l_sobel + 0.15 * l_freq + 0.1 * l_ssim
            loss_dict["charbonnier"] = l_charb.item()
            loss_dict["sobel"] = l_sobel.item()
            loss_dict["fourier"] = l_freq.item()
            loss_dict["ssim"] = l_ssim.item()

        else:
            # Configuration F: Full Composite Loss
            l_charb = self.charbonnier(pred, target)
            l_sobel = self.sobel(pred, target)
            l_freq = self.fourier(pred, target)
            l_ms_ssim = self.ms_ssim(pred, target)
            l_defect = self.defect(pred, target)
            total = l_charb + 0.2 * l_sobel + 0.15 * l_freq + 0.1 * l_ms_ssim + 0.2 * l_defect
            loss_dict["charbonnier"] = l_charb.item()
            loss_dict["sobel"] = l_sobel.item()
            loss_dict["fourier"] = l_freq.item()
            loss_dict["ms_ssim"] = l_ms_ssim.item()
            loss_dict["defect"] = l_defect.item()

        if router_logits is not None and target_deg_vector is not None:
            l_deg = self.deg_loss(router_logits, target_deg_vector)
            total = total + l_deg
            loss_dict["deg_loss"] = l_deg.item()

        loss_dict["total_loss"] = total.item()
        return total, loss_dict


if __name__ == "__main__":
    pred = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.randn(2, 1, 128, 128)
    for cfg in ["A", "B", "C", "D", "E", "F"]:
        loss_fn = V6AblationLoss(config_name=cfg)
        loss, details = loss_fn(pred, target)
        print(f"Config {cfg} Total Loss: {loss.item():.5f} | Details: {details}")
