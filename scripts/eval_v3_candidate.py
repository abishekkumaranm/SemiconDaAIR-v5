"""
eval_v3_candidate.py — Evaluates trained SemiconDaAIR-v3 Candidate Model against SemiconDaAIR-v2.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v2 import build_semicon_daair_v2
from models.semicon_daair_v3 import build_semicon_daair_v3
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim

V2_CKPT = "checkpoints/final/semicon_daair_v2_final.pt"
V3_CKPT = "checkpoints/final/semicon_daair_v3_candidate.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on {device}...")

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # 1. Evaluate v2
    v2_model = build_semicon_daair_v2(scale=2, base_channels=64).to(device)
    if os.path.exists(V2_CKPT):
        ckpt = torch.load(V2_CKPT, map_location=device)
        v2_model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt, strict=False)
    v2_model.eval()

    psnr_v2, ssim_v2 = [], []
    with torch.inference_mode():
        for lq_t, gt_t in val_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)
            out_hr = v2_model(lq_t)
            out_np = out_hr.squeeze(1).cpu().numpy()
            gt_np = gt_t.squeeze(1).cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_v2.append(compute_psnr(out_np[i], gt_np[i]))
                ssim_v2.append(compute_ssim(out_np[i], gt_np[i]))

    mean_p_v2 = float(np.mean(psnr_v2))
    mean_s_v2 = float(np.mean(ssim_v2))

    # 2. Evaluate v3 Candidate
    psnr_v3, ssim_v3 = [], []
    if os.path.exists(V3_CKPT):
        v3_model = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True).to(device)
        ckpt3 = torch.load(V3_CKPT, map_location=device)
        v3_model.load_state_dict(ckpt3["model_state"] if "model_state" in ckpt3 else ckpt3, strict=False)
        v3_model.eval()

        with torch.inference_mode():
            for lq_t, gt_t in val_loader:
                lq_t, gt_t = lq_t.to(device), gt_t.to(device)
                out_hr = v3_model(lq_t)
                if isinstance(out_hr, tuple):
                    out_hr = out_hr[0]
                out_np = out_hr.squeeze(1).cpu().numpy()
                gt_np = gt_t.squeeze(1).cpu().numpy()
                for i in range(out_np.shape[0]):
                    psnr_v3.append(compute_psnr(out_np[i], gt_np[i]))
                    ssim_v3.append(compute_ssim(out_np[i], gt_np[i]))

        mean_p_v3 = float(np.mean(psnr_v3))
        mean_s_v3 = float(np.mean(ssim_v3))
    else:
        mean_p_v3, mean_s_v3 = 0.0, 0.0

    print("=" * 70)
    print("      FINAL COMPETITION MODEL COMPARISON SUMMARY      ")
    print("=" * 70)
    print(f"SemiconDaAIR-v2 Baseline : PSNR = {mean_p_v2:.4f} dB | SSIM = {mean_s_v2:.4f}")
    print(f"SemiconDaAIR-v3 Candidate: PSNR = {mean_p_v3:.4f} dB | SSIM = {mean_s_v3:.4f}")
    print("=" * 70)

    # Write report
    lines = [
        "# SemiconDaAIR-v3 Final Candidate Model Selection Report\n",
        "**Target Competition**: KLA / SEMICON India Hackathon  \n",
        "**Evaluation Dataset**: 640 Paired Validation Samples (`splits/val.txt`)  \n",
        f"**Hardware Device**: {device}  \n\n",
        "---",
        "## 1. Quantitative Benchmark Results\n",
        "| Model Version | Checkpoint File | Parameters | Val PSNR (dB) | Val SSIM | Selection Status |",
        "|---|---|---|---|---|---|",
        f"| **`SemiconDaAIR-v2`** | `checkpoints/final/semicon_daair_v2_final.pt` | 544,628 | **{mean_p_v2:.4f}** | **{mean_s_v2:.4f}** | Gold Standard Baseline |",
        f"| **`SemiconDaAIR-v3`** | `checkpoints/final/semicon_daair_v3_candidate.pt` | 605,744 | **{mean_p_v3:.4f}** | **{mean_s_v3:.4f}** | Upgraded Candidate |",
        "\n---",
        "## 2. Server Status & REST API Activation\n",
        f"Both models are available in `serve.py`. When `checkpoints/final/semicon_daair_v3_candidate.pt` exists, `serve.py` automatically loads **`SemiconDaAIR-v3` as the DEFAULT PRIMARY MODEL** on `http://127.0.0.1:8000/`.\n"
    ]

    with open("reports/v3_model_selection.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved reports/v3_model_selection.md successfully!")


if __name__ == "__main__":
    main()
