"""
preprocessing.py — Exact Image Loading, Formatting & Preprocessing Pipeline.

Rules:
  - Preserves exact floating-point detector values for .npy files without blind /255 clamping.
  - Converts 8-bit/16-bit PNG/TIFF/JPG images cleanly to single-channel float32 [1, 1, H, W] tensors.
  - Preserves 1-channel grayscale representation.
  - Matches exact training preprocessing.
"""

import os
import cv2
import numpy as np
import torch


def load_image_exact(file_path: str) -> np.ndarray:
    """
    Loads an image from file_path matching exact dataset ingestion:
      - .npy: Loads raw float32 array directly
      - Image formats (.png, .jpg, .tif): Converts to 1-channel grayscale float32 normalized in [0, 1]
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return arr

    # Standard image format
    img_bgr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Unable to read image file: {file_path}")

    if img_bgr.ndim == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    # Handle uint16 vs uint8
    if img_gray.dtype == np.uint16:
        arr_float = img_gray.astype(np.float32) / 65535.0
    elif img_gray.dtype == np.uint8:
        arr_float = img_gray.astype(np.float32) / 255.0
    else:
        arr_float = img_gray.astype(np.float32)

    return arr_float


def save_image_exact(img_np: np.ndarray, output_path: str):
    """
    Saves restored array to output_path:
      - Saves .npy raw float32 array if requested
      - Saves visual 8-bit PNG/JPG/TIFF image with percentile display scaling
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".npy":
        np.save(output_path, img_np.astype(np.float32))
        return

    # Visual display normalization
    if img_np.min() < 0.0 or img_np.max() > 1.0:
        p_min, p_max = np.percentile(img_np, (0.5, 99.5))
        if p_max > p_min:
            img_norm = np.clip((img_np - p_min) / (p_max - p_min), 0.0, 1.0)
        else:
            img_norm = np.clip(img_np, 0.0, 1.0)
    else:
        img_norm = np.clip(img_np, 0.0, 1.0)

    img_uint8 = (img_norm * 255.0).round().astype(np.uint8)
    cv2.imwrite(output_path, img_uint8)
