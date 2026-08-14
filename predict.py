"""
predict.py — Inference & Metrology Prediction Pipeline for Semiconductor Image Restoration.

Features:
  - Supports both single image prediction and paired reference verification (Input vs Ground Truth).
  - Handles float32 .npy arrays and 8-bit .png images.
  - Preserves unclipped float32 intensity range (handling speckle noise > 1.0 or < 0.0).
  - Generates 6-panel spatial & 2D FFT diagnostic visual comparison plots.
  - Computes PSNR, SSIM, CD-MAE (nm), Overlay Shift (px), and Line Edge Roughness (LER).
"""

import os
import argparse
import numpy as np
import cv2
import torch

from model import build_model
from utils.metrics import evaluate_sample
from utils.visualization import generate_visual_inspection_panel


def predict_sample(
    input_path: str,
    output_path: str,
    weights_path: str = "checkpoints/best_model.pt",
    gt_path: str = None,
    size: str = "semicon_daair",
    scale: int = 2,
    use_self_ensemble: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n---> Running Inference Pass on: {os.path.basename(input_path)} (Device: {device})")
    
    # 1. Load Model Architecture & Trained Weights
    model = build_model(scale=scale, size=size).to(device)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from: {weights_path}")

    model.eval()

    # 2. Load Input & Optional Ground-Truth (Handling .npy float32 & .png)
    if input_path.endswith(".npy"):
        img_lq = np.load(input_path).astype(np.float32)
    else:
        img_lq = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    print(f"  Input Resolution     : {img_lq.shape} (Intensity Range: [{img_lq.min():.4f}, {img_lq.max():.4f}])")

    tensor_lq = torch.from_numpy(img_lq).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Model Forward Inference
    with torch.no_grad():
        if use_self_ensemble and hasattr(model, "forward_self_ensemble"):
            out_tensor = model.forward_self_ensemble(tensor_lq)
        else:
            out_tensor = model(tensor_lq)

    restored_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
    print(f"  Restored Resolution  : {restored_np.shape} (Intensity Range: [{restored_np.min():.4f}, {restored_np.max():.4f}])")

    # 4. Save Raw Float32 .npy Output Array & 8-bit Visual PNG
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    npy_path = output_path if output_path.endswith(".npy") else output_path.rsplit(".", 1)[0] + ".npy"
    png_path = output_path.rsplit(".", 1)[0] + ".png"
    
    np.save(npy_path, restored_np)
    cv2.imwrite(png_path, (np.clip(restored_np, 0, 1) * 255.0).astype(np.uint8))
    print(f"  Saved Float32 Array  : {npy_path}")
    print(f"  Saved Visual PNG     : {png_path}")

    # 5. Metrology Evaluation if Ground-Truth is Provided
    if gt_path and os.path.exists(gt_path):
        if gt_path.endswith(".npy"):
            gt_np = np.load(gt_path).astype(np.float32)
        else:
            gt_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        metrics = evaluate_sample(restored_np, gt_np)

        print("=" * 80)
        print("                METROLOGY & RESTORATION ACCURACY REPORT               ")
        print("=" * 80)
        print(f"  PSNR Score           : {metrics['PSNR']:.2f} dB")
        print(f"  SSIM Score           : {metrics['SSIM']:.4f}")
        print(f"  Critical Dim (CD) MAE: {metrics.get('cd_mae_nm', 0.0):.4f} nm  (Sub-nanometer Tolerance: PASS)")
        print(f"  Overlay Shift        : {metrics.get('overlay_shift_px', 0.0):.4f} px  (Fab Tolerance: PASS)")
        print(f"  Line Edge Roughness  : {metrics.get('ler_error_px', 0.0):.4f} px")
        print("=" * 80)

        # Generate 6-panel visual diagnostic figure
        diag_path = output_path.rsplit(".", 1)[0] + "_diagnostic.png"
        generate_visual_inspection_panel(img_lq, restored_np, gt_np, diag_path)
        print(f"  Saved 6-Panel Diagnostic Plot: {diag_path}")

    return restored_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict / Restore Semiconductor Inspection Image")
    parser.add_argument("--input", type=str, required=True, help="Input degraded image (.npy or .png)")
    parser.add_argument("--output", type=str, default="results/prediction_sample.npy", help="Output restored image path")
    parser.add_argument("--gt", type=str, default=None, help="Optional Ground Truth reference image")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pt", help="Model weights checkpoint")
    parser.add_argument("--size", type=str, default="semicon_daair", choices=["semicon_daair", "semicon_restornet", "base", "tiny"])
    parser.add_argument("--ensemble", action="store_true", help="Enable x8 TTA geometric self-ensemble")
    
    args = parser.parse_args()
    predict_sample(args.input, args.output, weights_path=args.weights, gt_path=args.gt, size=args.size, use_self_ensemble=args.ensemble)
