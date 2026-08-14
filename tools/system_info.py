"""
system_info.py — Automatic Laptop Hardware & Environment Diagnostic Tool (Pure Standard Library).

Detects and reports:
  - Operating System & Architecture
  - Python & PyTorch environment details
  - CPU cores
  - NVIDIA GPU, VRAM, CUDA runtime
  - Generates reports/hardware_report.md
"""

import os
import sys
import platform
import torch


def detect_system_info():
    os_name = platform.system()
    os_version = platform.version()
    os_arch = platform.machine()
    python_ver = platform.python_version()

    cpu_name = platform.processor() or "Generic x86_64 CPU"
    cpu_cores_logical = os.cpu_count() or 4

    torch_ver = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_available else "N/A"

    gpu_name = "N/A (CPU Only)"
    vram_gb = 0.0
    recommended_device = "cpu"

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        recommended_device = "cuda"

    info = {
        "os": f"{os_name} {os_arch} ({os_version})",
        "python_version": python_ver,
        "pytorch_version": torch_ver,
        "cpu": f"{cpu_name} ({cpu_cores_logical} logical cores)",
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "cuda_available": cuda_available,
        "cuda_version": cuda_ver,
        "recommended_device": recommended_device
    }

    # Print to console
    print("=" * 65)
    print("           SemiconDaAIR SYSTEM HARDWARE CHECK          ")
    print("=" * 65)
    print(f"OS                 : {info['os']}")
    print(f"Python             : {info['python_version']}")
    print(f"PyTorch            : {info['pytorch_version']}")
    print(f"CPU                : {info['cpu']}")
    print(f"GPU                : {info['gpu']}")
    print(f"VRAM               : {info['vram_gb']} GB")
    print(f"CUDA Available     : {info['cuda_available']}")
    print(f"CUDA Version       : {info['cuda_version']}")
    print(f"Recommended Device : {info['recommended_device']}")
    print("=" * 65)

    # Save to reports/hardware_report.md
    os.makedirs("reports", exist_ok=True)
    report_md = f"""# 💻 SemiconDaAIR Hardware Diagnostic Report

## ⚙️ System Environment
- **Operating System**: {info['os']}
- **Python Version**: {info['python_version']}
- **PyTorch Version**: {info['pytorch_version']}
- **CPU**: {info['cpu']}

## 🎮 GPU Acceleration & CUDA
- **GPU Device**: {info['gpu']}
- **VRAM Total Memory**: {info['vram_gb']} GB
- **CUDA Available**: `{info['cuda_available']}`
- **CUDA Version**: `{info['cuda_version']}`
- **Recommended PyTorch Device**: `{info['recommended_device']}`

---
*Generated automatically by `tools/system_info.py`*
"""
    with open("reports/hardware_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("Saved Hardware Report to: reports/hardware_report.md\n")
    return info


if __name__ == "__main__":
    detect_system_info()
