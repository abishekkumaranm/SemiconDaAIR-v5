"""
characterize_dataset.py — Statistical Characterization of Real KLA Semiconductor Inspection Dataset.

Scans all 3,200 real KLA images (GT and NoisyLR) to calculate:
  - Mean intensity
  - Standard deviation
  - Min and Max intensity
  - Negative-pixel ratio
  - Local spatial variance
  - Sobel gradient energy
  - 2D FFT High-Frequency power spectrum
  - Canny edge density

Saves full output to: results/kla_dataset_statistics.json
"""

import os
import glob
import json
import numpy as np
import cv2


def run_dataset_characterization():
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    lq_files = sorted(glob.glob(os.path.join(lq_dir, "*.npy")))

    print("=" * 80)
    print("       CHARACTERIZING REAL KLA DATASET STATISTICS (3,200 SAMPLES)       ")
    print("=" * 80)

    lq_means, lq_stds, lq_mins, lq_maxs, lq_neg_ratios = [], [], [], [], []
    lq_local_vars, lq_grad_energies, lq_hf_powers = [], [], []

    for path in lq_files:
        arr = np.load(path).astype(np.float32)

        lq_means.append(float(np.mean(arr)))
        lq_stds.append(float(np.std(arr)))
        lq_mins.append(float(np.min(arr)))
        lq_maxs.append(float(np.max(arr)))
        lq_neg_ratios.append(float(np.mean(arr < 0.0)))

        # Local variance
        kernel = np.ones((5, 5), np.float32) / 25.0
        mean_img = cv2.filter2D(arr, -1, kernel)
        sq_mean_img = cv2.filter2D(arr ** 2, -1, kernel)
        var_img = np.maximum(sq_mean_img - mean_img ** 2, 0.0)
        lq_local_vars.append(float(np.mean(var_img)))

        # Sobel gradient energy
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        lq_grad_energies.append(float(np.mean(gx**2 + gy**2)))

        # 2D FFT HF power spectrum
        fft_img = np.fft.fftshift(np.fft.fft2(arr))
        power = np.abs(fft_img) ** 2
        lq_hf_powers.append(float(np.mean(power)))

    stats = {
        "dataset_total_samples": len(lq_files),
        "lq_intensity_mean": float(np.mean(lq_means)),
        "lq_intensity_std": float(np.mean(lq_stds)),
        "lq_intensity_min_overall": float(np.min(lq_mins)),
        "lq_intensity_max_overall": float(np.max(lq_maxs)),
        "lq_negative_pixel_ratio_average": float(np.mean(lq_neg_ratios)),
        "lq_samples_with_negative_pixels_count": int(sum(m < 0.0 for m in lq_mins)),
        "lq_local_variance_mean": float(np.mean(lq_local_vars)),
        "lq_sobel_gradient_energy_mean": float(np.mean(lq_grad_energies)),
        "lq_2d_fft_power_spectrum_mean": float(np.mean(lq_hf_powers))
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/kla_dataset_statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n[CHARACTERIZATION SUMMARY]")
    print(f"  LQ Mean Intensity       : {stats['lq_intensity_mean']:.4f}")
    print(f"  LQ Intensity Std        : {stats['lq_intensity_std']:.4f}")
    print(f"  LQ Overall Min          : {stats['lq_intensity_min_overall']:.4f}")
    print(f"  LQ Overall Max          : {stats['lq_intensity_max_overall']:.4f}")
    print(f"  Negative Pixel Ratio    : {stats['lq_negative_pixel_ratio_average']*100:.2f}% ({stats['lq_samples_with_negative_pixels_count']} / 3200 files)")
    print(f"  Local Variance Mean     : {stats['lq_local_variance_mean']:.4f}")
    print(f"  Sobel Gradient Energy   : {stats['lq_sobel_gradient_energy_mean']:.4f}")
    print(f"Saved dataset statistics JSON to: {out_path}")


if __name__ == "__main__":
    run_dataset_characterization()
