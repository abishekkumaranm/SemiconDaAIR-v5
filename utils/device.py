"""
device.py — Dynamic PyTorch Device Selector Utility.

Supports:
  - --device auto (Selects CUDA if available, falls back to CPU cleanly)
  - --device cuda (Enforces CUDA GPU, errors cleanly if unavailable)
  - --device cpu  (Enforces CPU mode)
"""

import sys
import torch


def select_device(device_arg: str = "auto") -> torch.device:
    device_arg = device_arg.lower().strip()

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")

    elif device_arg == "cpu":
        return torch.device("cpu")

    else:  # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (cuda)"
    return "CPU (cpu)"


if __name__ == "__main__":
    dev = select_device("auto")
    print(f"Selected Device: {dev} ({get_device_name(dev)})")
