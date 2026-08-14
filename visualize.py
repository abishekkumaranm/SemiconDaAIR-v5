"""
visualize.py — Generates Visual Inspection Comparison Artifacts:
  - Input Degraded Image
  - Restored Output Image
  - Ground Truth Reference Image
  - Absolute Difference Map (|Restored - GT|)
  - Sobel Edge Gradient Map
  - 2D Fourier Magnitude Spectrum
"""

import os
import glob
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from model import build_model
from degrade import degrade_image


def generate_visualizations(clean_dir="data/clean_images", output_dir="results/visual_inspection", weights_path="checkpoints/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(scale=2, size="semicon_restornet").to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(clean_dir, "*.png")))[:4]

    for idx, path in enumerate(image_paths):
        gt_uint8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        lq_float, scale = degrade_image(gt_uint8, scale=2)
        gt_float = gt_uint8.astype(np.float32) / 255.0

        tensor_lq = torch.from_numpy(lq_float).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            tensor_out = model(tensor_lq)

        restored_float = tensor_out.squeeze(0).squeeze(0).cpu().numpy()

        # Resize LQ for visual alignment
        lq_resized = cv2.resize(lq_float, (gt_float.shape[1], gt_float.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Difference map
        diff_map = np.abs(restored_float - gt_float)

        # Edge maps
        sobel_gt = cv2.Sobel(gt_float, cv2.CV_32F, 1, 1)
        sobel_restored = cv2.Sobel(restored_float, cv2.CV_32F, 1, 1)

        # Fourier Spectrums
        fft_gt = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(gt_float))))
        fft_restored = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(restored_float))))

        # Plot 2x3 Figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(lq_resized, cmap="gray")
        axes[0, 0].set_title("Input Degraded (Speckle+Blur+2x Down)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(restored_float, cmap="gray")
        axes[0, 1].set_title("SemiconRestorNet Output")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(gt_float, cmap="gray")
        axes[0, 2].set_title("Ground Truth Reference")
        axes[0, 2].axis("off")

        im_diff = axes[1, 0].imshow(diff_map, cmap="jet")
        axes[1, 0].set_title("Absolute Error Map |Restored - GT|")
        axes[1, 0].axis("off")
        plt.colorbar(im_diff, ax=axes[1, 0], fraction=0.046)

        axes[1, 1].imshow(sobel_restored, cmap="magma")
        axes[1, 1].set_title("Restored Edge Gradient")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(fft_restored, cmap="viridis")
        axes[1, 2].set_title("Restored 2D Fourier Spectrum")
        axes[1, 2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"visual_inspection_sample_{idx}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved visual comparison artifact: {save_path}")


if __name__ == "__main__":
    generate_visualizations()
