"""
test_api_restore.py — Integration Test for /api/restore Endpoint.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serve import load_array_from_bytes, tiled_inference, MODEL_V2, DEVICE

sample_file = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR\000068.npy"

if os.path.exists(sample_file):
    with open(sample_file, "rb") as f:
        content = f.read()

    input_arr, dtype_str = load_array_from_bytes(os.path.basename(sample_file), content)
    print(f"Loaded Array -> shape: {input_arr.shape}, dtype: {dtype_str}")

    tensor_in = torch.from_numpy(input_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    out_tensor, router_dict = tiled_inference(MODEL_V2, tensor_in, scale=2, patch_size=128, stride=64, device=DEVICE)

    output_arr = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
    print(f"[PASS] Inference Succeeded -> Output shape: {output_arr.shape}, Range: [{np.min(output_arr):.4f}, {np.max(output_arr):.4f}]")
    print(f"Router Dict: {router_dict}")
else:
    print(f"Sample file not found at {sample_file}")
