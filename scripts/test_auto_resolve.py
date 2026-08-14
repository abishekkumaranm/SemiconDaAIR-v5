"""
test_auto_resolve.py — Test macOS shadow file auto-resolution in serve.py.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serve import load_array_from_bytes

dummy_mac_filename = "._000040.npy"
dummy_mac_content = b"macOS shadow header"

try:
    arr, dtype_str = load_array_from_bytes(dummy_mac_filename, dummy_mac_content)
    print(f"[SUCCESS] Auto-resolved {dummy_mac_filename} -> Loaded shape: {arr.shape}, range: [{arr.min():.4f}, {arr.max():.4f}]")
except Exception as e:
    print(f"[FAIL] {e}")
