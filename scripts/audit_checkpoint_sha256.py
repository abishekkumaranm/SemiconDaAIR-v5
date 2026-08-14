"""
audit_checkpoint_sha256.py — Audit best_psnr.pt checkpoint SHA256, file size, parameter count,
and create protected final checkpoint copy at checkpoints/final/semicon_daair_v2_final.pt.
"""

import os
import hashlib
import shutil
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v2 import build_semicon_daair_v2


def audit_checkpoint():
    ckpt_path = os.path.abspath("checkpoints/exp02/best_psnr.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!")
        return

    # 1. File Size
    file_size_bytes = os.path.getsize(ckpt_path)

    # 2. SHA256 Hash
    sha256_hash = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    digest = sha256_hash.hexdigest()

    # 3. Model Parameter Count
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_semicon_daair_v2(scale=2, base_channels=64)
    param_count = sum(p.numel() for p in model.parameters())

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    print("=" * 60)
    print("      SEMICONDAAIR-V2 CHECKPOINT AUDIT SUMMARY      ")
    print("=" * 60)
    print(f"Checkpoint File Path : {ckpt_path}")
    print(f"File Size (Bytes)   : {file_size_bytes:,} bytes ({file_size_bytes / (1024**2):.2f} MB)")
    print(f"SHA256 Hash         : {digest}")
    print(f"Model Parameter Count: {param_count:,}")
    print(f"Target Architecture  : SemiconDaAIR-v2")

    # 4. Create Protected Final Copy
    final_dir = os.path.abspath("checkpoints/final")
    os.makedirs(final_dir, exist_ok=True)
    final_ckpt_path = os.path.join(final_dir, "semicon_daair_v2_final.pt")
    
    if not os.path.exists(final_ckpt_path):
        shutil.copy2(ckpt_path, final_ckpt_path)
        print(f"Created Protected Final Copy at: {final_ckpt_path}")
    else:
        print(f"Protected Final Copy already exists at: {final_ckpt_path}")

    # Audit log report
    audit_data = {
        "checkpoint_path": ckpt_path,
        "final_copy_path": final_ckpt_path,
        "file_size_bytes": file_size_bytes,
        "sha256": digest,
        "parameter_count": param_count,
        "architecture": "SemiconDaAIR-v2"
    }
    
    with open("results/checkpoint_audit.json", "w") as f:
        import json
        json.dump(audit_data, f, indent=4)
    print("Saved audit log to: results/checkpoint_audit.json")


if __name__ == "__main__":
    audit_checkpoint()
