"""
test_tiled_inference.py — Test Tiled Overlapping Patch Inference for Full-Resolution Images.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v2 import build_semicon_daair_v2
from models.semicon_daair_v3 import build_semicon_daair_v3


def tiled_inference(model, tensor_in, scale=2, patch_size=128, stride=96, device="cuda"):
    """
    Overlapping patch-based inference for arbitrary resolution input images.
    Input tensor_in: [1, 1, H, W]
    Returns restored tensor: [1, 1, scale*H, scale*W]
    """
    _, _, h, w = tensor_in.shape
    
    if h <= patch_size and w <= patch_size:
        with torch.inference_mode():
            return model(tensor_in)

    out_h, out_w = h * scale, w * scale
    patch_out_size = patch_size * scale
    stride_out = stride * scale

    output = torch.zeros((1, 1, out_h, out_w), device=device, dtype=torch.float32)
    weights = torch.zeros((1, 1, out_h, out_w), device=device, dtype=torch.float32)

    # 2D Hanning window for smooth patch blending
    window_1d = torch.hann_window(patch_out_size, periodic=False, device=device)
    window_2d = (window_1d.unsqueeze(1) @ window_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    y_steps = list(range(0, max(1, h - patch_size + 1), stride))
    if y_steps[-1] + patch_size < h:
        y_steps.append(h - patch_size)

    x_steps = list(range(0, max(1, w - patch_size + 1), stride))
    if x_steps[-1] + patch_size < w:
        x_steps.append(w - patch_size)

    with torch.inference_mode():
        for y in y_steps:
            for x in x_steps:
                patch = tensor_in[:, :, y:y+patch_size, x:x+patch_size]
                out_patch = model(patch)
                if isinstance(out_patch, tuple):
                    out_patch = out_patch[0]

                oy, ox = y * scale, x * scale
                output[:, :, oy:oy+patch_out_size, ox:ox+patch_out_size] += out_patch * window_2d
                weights[:, :, oy:oy+patch_out_size, ox:ox+patch_out_size] += window_2d

    weights = torch.clamp(weights, min=1e-5)
    return output / weights


def main():
    img_path = r"C:\Users\HP\Downloads\image.png"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    raw_img = Image.open(img_path).convert("L")
    in_arr = np.array(raw_img, dtype=np.float32) / 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v2 = build_semicon_daair_v2(scale=2, base_channels=64).to(device)

    ckpt_path = "checkpoints/final/semicon_daair_v2_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model_v2.load_state_dict(state_dict, strict=False)
    model_v2.eval()

    tensor_in = torch.from_numpy(in_arr).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Input image shape: {tensor_in.shape}")
    print("Running Tiled Overlapping Patch Inference (patch=128, stride=64)...")

    t0 = os.times().elapsed
    out_tiled = tiled_inference(model_v2, tensor_in, scale=2, patch_size=128, stride=64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = os.times().elapsed

    out_arr = out_tiled.squeeze(0).squeeze(0).cpu().numpy()

    print(f"Tiled Inference Complete in {(t1 - t0)*1000.0:.2f} ms!")
    print(f"Restored Output Shape: {out_arr.shape}")
    print(f"Output Dynamic Range: [{np.min(out_arr):.4f}, {np.max(out_arr):.4f}]")

    out_dir = "results/tiled_restoration"
    os.makedirs(out_dir, exist_ok=True)
    out_png_path = os.path.join(out_dir, "tiled_image_restored.png")

    norm_out = np.clip((out_arr - np.min(out_arr)) / (np.max(out_arr) - np.min(out_arr) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(norm_out).save(out_png_path)
    print(f"Saved Tiled Restored Image to: {out_png_path}")


if __name__ == "__main__":
    main()
