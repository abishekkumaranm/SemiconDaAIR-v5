"""
eval_with_metrics.py — Metrology & Image Quality Evaluation Benchmark Script.

Computes PSNR, SSIM, RMSE, MAE, Critical Dimension (CD) Error, Overlay Shift Error, and Line Edge Roughness (LER)
against Ground-Truth reference images. Outputs summary metrics JSON and comparison figures.
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import torch

from model import build_model
from metrics import evaluate_sample


def evaluate_val_set(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Executing self-evaluation benchmark on device: {device}")

    model = build_model(scale=args.scale, size=args.size).to(device)
    if os.path.exists(args.weights):
        print(f"Loading weights from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    gt_paths = sorted(
        glob.glob(os.path.join(args.gt_dir, "*.npy")) +
        glob.glob(os.path.join(args.gt_dir, "*.png")) +
        glob.glob(os.path.join(args.gt_dir, "*.jpg"))
    )
    if not gt_paths:
        raise RuntimeError(f"No ground-truth images found in: {args.gt_dir}")

    all_metrics = []

    with torch.no_grad():
        for path in gt_paths:
            filename = os.path.basename(path)
            base_name, ext = os.path.splitext(filename)

            if path.endswith(".npy"):
                gt_img = np.load(path).astype(np.float32)
            else:
                gt_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

            lq_path = os.path.join(args.lq_dir, filename) if args.lq_dir else None
            if lq_path and os.path.exists(lq_path):
                if lq_path.endswith(".npy"):
                    lq_img = np.load(lq_path).astype(np.float32)
                else:
                    lq_img = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            else:
                from degrade import degrade_image
                lq_img, _ = degrade_image((np.clip(gt_img, 0, 1) * 255.0).astype(np.uint8), scale=args.scale)

            tensor_lq = torch.from_numpy(lq_img).unsqueeze(0).unsqueeze(0).to(device)
            tensor_out = model(tensor_lq)
            restored_img = tensor_out.squeeze(0).squeeze(0).cpu().numpy()

            metrics = evaluate_sample(restored_img, gt_img)
            metrics["filename"] = filename
            all_metrics.append(metrics)

            vis_lq = cv2.resize(lq_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            comp = np.hstack([vis_lq, np.clip(restored_img, 0, 1), gt_img])
            comp_uint8 = (comp * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, f"comp_{base_name}.png"), comp_uint8)

    summary = {
        "num_samples": len(all_metrics),
        "mean_PSNR": float(np.mean([m["PSNR"] for m in all_metrics])),
        "mean_SSIM": float(np.mean([m["SSIM"] for m in all_metrics])),
        "mean_RMSE": float(np.mean([m["RMSE"] for m in all_metrics])),
        "mean_MAE": float(np.mean([m["MAE"] for m in all_metrics])),
        "mean_CD_MAE_nm": float(np.mean([m["cd_mae_nm"] for m in all_metrics])),
        "mean_Overlay_Shift_px": float(np.mean([m["overlay_shift_px"] for m in all_metrics])),
        "mean_LER_Error_px": float(np.mean([m["ler_error_px"] for m in all_metrics])),
    }

    print("\n=== METROLOGY & RESTORATION BENCHMARK SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved metrics summary JSON to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SemiconRestorNet against Ground Truth Validation Set")
    parser.add_argument("--gt_dir", type=str, default="data/clean_images", help="Ground truth images directory")
    parser.add_argument("--lq_dir", type=str, default=None, help="Degraded images directory (optional)")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pt", help="Model weights path")
    parser.add_argument("--output_dir", type=str, default="results/self_eval", help="Output directory")
    parser.add_argument("--size", type=str, default="semicon_restornet", choices=["semicon_restornet", "base", "tiny"])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    evaluate_val_set(args)
