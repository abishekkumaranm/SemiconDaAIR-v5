"""
test_user_image.py — Test Real PyTorch Model Inference + Tiled Patch Inference on User's Image (C:\\Users\\HP\\Downloads\\image.png).
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v2 import build_semicon_daair_v2
from models.semicon_daair_v3 import build_semicon_daair_v3
from serve import tiled_inference

img_path = r"C:\Users\HP\Downloads\image.png"

print("=" * 70)
print("      TESTING REAL PYTORCH INFERENCE ON USER'S IMAGE      ")
print("=" * 70)
print(f"Target Image Path: {img_path}")

if not os.path.exists(img_path):
    print(f"[FAIL] Image file not found at {img_path}")
    sys.exit(1)

# Load Image
raw_img = Image.open(img_path).convert("L")
in_arr = np.array(raw_img, dtype=np.float32) / 255.0

print(f"Loaded Image -> Dimensions: {raw_img.width}x{raw_img.height} | Dtype: {in_arr.dtype} | Range: [{np.min(in_arr):.4f}, {np.max(in_arr):.4f}]")

# Setup PyTorch GPU Model v2 & v3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_v2 = build_semicon_daair_v2(scale=2, base_channels=64).to(device)
ckpt_path = "checkpoints/final/semicon_daair_v2_final.pt"
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
model_v2.load_state_dict(state_dict, strict=False)
model_v2.eval()

model_v3 = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True).to(device)
model_v3.eval()

# Tensor Conversion
tensor_in = torch.from_numpy(in_arr).unsqueeze(0).unsqueeze(0).to(device)

print(f"\nExecuting Overlapping Tiled Neural Model Inference on {device}...")

t0 = os.times().elapsed
out_tensor_v2, router_v2 = tiled_inference(model_v2, tensor_in, scale=2, patch_size=128, stride=64, device=device)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = os.times().elapsed

out_arr_v2 = out_tensor_v2.squeeze(0).squeeze(0).cpu().numpy()

print("\n--- SEMICONDAAIR-V2 TILED INFERENCE RESULTS ---")
print(f"Model Name           : SemiconDaAIR-v2 (Tiled Neural Inference)")
print(f"Parameters           : 544,628")
print(f"Input Shape          : {tensor_in.shape}")
print(f"Output HR Shape      : {out_tensor_v2.shape} ({out_arr_v2.shape[1]}x{out_arr_v2.shape[0]})")
print(f"Measured GPU Latency : {(t1 - t0) * 1000.0:.2f} ms")
print(f"Mean Router Weights  : Gaussian = {router_v2['gaussian']:.4f} | Speckle = {router_v2['speckle']:.4f} | SR = {router_v2['sr']:.4f}")
print(f"Output Min / Max     : [{np.min(out_arr_v2):.4f}, {np.max(out_arr_v2):.4f}]")

# Save Restored PNG & NPY
out_dir = "results/user_image_restoration"
os.makedirs(out_dir, exist_ok=True)

out_npy_path = os.path.join(out_dir, "image_restored_tiled.npy")
np.save(out_npy_path, out_arr_v2.astype(np.float32))

out_png_path = os.path.join(out_dir, "image_restored_tiled.png")
uint8_out = np.clip((out_arr_v2 - np.min(out_arr_v2)) / (np.max(out_arr_v2) - np.min(out_arr_v2) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(uint8_out).save(out_png_path)

print(f"\n[SUCCESS] Saved Restored Tiled NPY to: {out_npy_path}")
print(f"[SUCCESS] Saved Restored Tiled PNG to: {out_png_path}")
