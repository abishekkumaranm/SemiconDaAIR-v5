"""
debug_nan.py — Pinpoint exact NaN source in SemiconDaAIR-v2 forward/loss pass.
"""

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.semicon_daair_v2 import build_semicon_daair_v2
from losses import RestorationLoss


def debug_nan_pass():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_semicon_daair_v2(scale=2, base_channels=64).to(device)
    criterion = RestorationLoss().to(device)
    
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    lq_path = os.path.join(lq_dir, "000000.npy")
    gt_path = os.path.join(gt_dir, "000000.npy")

    import numpy as np
    lq = torch.from_numpy(np.load(lq_path)).unsqueeze(0).unsqueeze(0).to(device)
    gt = torch.from_numpy(np.load(gt_path)).unsqueeze(0).unsqueeze(0).to(device)

    print("Input LQ stats: min=", lq.min().item(), "max=", lq.max().item(), "has_nan=", torch.isnan(lq).any().item())
    print("GT stats: min=", gt.min().item(), "max=", gt.max().item(), "has_nan=", torch.isnan(gt).any().item())

    # Forward
    pred, logits = model(lq, return_router_logits=True)
    print("Pred stats: min=", pred.min().item(), "max=", pred.max().item(), "has_nan=", torch.isnan(pred).any().item())

    loss, details = criterion(pred, gt, router_logits=logits, target_deg_vector=torch.ones((1, 3), device=device))
    print("Loss details:", details)
    
    loss.backward()
    print("Backward pass completed without anomaly error.")


if __name__ == "__main__":
    debug_nan_pass()
