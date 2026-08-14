"""
debug_api_upload.py — Debug Script to test multipart form parsing and array loading.
"""

import os
import io
import numpy as np

sample_file = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR\000068.npy"

if os.path.exists(sample_file):
    with open(sample_file, "rb") as f:
        content = f.read()

    buf = io.BytesIO(content)
    arr = np.load(buf).astype(np.float32)
    print(f"[SUCCESS] Loaded {sample_file} -> shape: {arr.shape}, dtype: {arr.dtype}, range: [{np.min(arr):.4f}, {np.max(arr):.4f}]")
else:
    print(f"Sample file not found at {sample_file}")
