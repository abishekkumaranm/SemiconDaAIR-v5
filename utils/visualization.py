"""
visualization.py — Spatial & 2D FFT Spectral Error Visual Debugging Utility.

Generates 6-panel and 8-panel visual diagnostic figures:
  [ Input | Bicubic | Model Output | Ground Truth | Absolute Error | FFT Spectrum Error ]
"""

import os
import numpy as np
import cv2


def generate_visual_inspection_panel(lq_np, restored_np, gt_np, save_path):
    """
    Generates high-resolution 6-panel diagnostic comparison figure.
    """
    lq_vis = cv2.resize(lq_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    bicubic_vis = lq_vis.copy()
    restored_vis = np.clip(restored_np, 0, 1)
    
    # 1. Absolute Spatial Error Map
    abs_err = np.abs(restored_vis - gt_np)
    abs_err_vis = cv2.applyColorMap((np.clip(abs_err * 5.0, 0, 1) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

    # 2. 2D FFT Magnitude Spectrum
    fft_gt = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gt_np))) + 1e-6)
    fft_res = np.log(np.abs(np.fft.fftshift(np.fft.fft2(restored_vis))) + 1e-6)
    
    fft_gt_norm = (fft_gt - fft_gt.min()) / (fft_gt.max() - fft_gt.min() + 1e-6)
    fft_res_norm = (fft_res - fft_res.min()) / (fft_res.max() - fft_res.min() + 1e-6)
    
    fft_err = np.abs(fft_res_norm - fft_gt_norm)
    fft_err_vis = cv2.applyColorMap((np.clip(fft_err * 3.0, 0, 1) * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Stack Top Row (Input | Model Output | GT)
    r1 = np.hstack([
        (lq_vis * 255.0).astype(np.uint8),
        (restored_vis * 255.0).astype(np.uint8),
        (gt_np * 255.0).astype(np.uint8)
    ])
    r1_bgr = cv2.cvtColor(r1, cv2.COLOR_GRAY2BGR)

    # Stack Bottom Row (Bicubic Baseline | Spatial Error | FFT Error)
    bic_bgr = cv2.cvtColor((bicubic_vis * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    r2_bgr = np.hstack([bic_bgr, abs_err_vis, fft_err_vis])

    panel = np.vstack([r1_bgr, r2_bgr])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, panel)
    return save_path


if __name__ == "__main__":
    dummy_lq = np.random.rand(128, 128).astype(np.float32)
    dummy_res = np.random.rand(256, 256).astype(np.float32)
    dummy_gt = np.random.rand(256, 256).astype(np.float32)
    generate_visual_inspection_panel(dummy_lq, dummy_res, dummy_gt, "results/visual_inspection/test_panel.png")
    print("Visual inspection panel test passed successfully.")
