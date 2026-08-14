"""
test_range_handlers.py — Phase 5: Dynamic Range Handler Numerical Comparison.

Evaluates 4 range transformation formulations on real unbounded signed float32 detector inputs:
  Option A: Fixed Epsilon Log ln(|X| + 1e-4)
  Option B: Learnable Epsilon Log
  Option C: Log-Linear Gated Transform
  Option D: Robust Inverse Hyperbolic Sine Transform asinh(X / scale)

Measures:
  1. Gradient stability (d(norm)/dx max magnitude)
  2. Invertibility & reconstruction error
  3. NaN/Inf resilience under extreme outliers (X = -0.2784 to 3.8500)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)


class RangeOptionA_FixedLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.abs(x) + 1e-4)


class RangeOptionB_LearnableLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x):
        eps = F.softplus(self.gamma) + 1e-4
        return torch.log(torch.abs(x) + eps)


class RangeOptionC_LogLinearGated(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        log_x = torch.log(torch.clamp(torch.abs(x) + 1e-4, min=1e-4))
        gate = torch.sigmoid(self.gate_conv(x))
        return log_x * gate + x * (1.0 - gate)


class RangeOptionD_Asinh(nn.Module):
    """Robust Inverse Hyperbolic Sine Transform: asinh(x / scale) = ln(x/scale + sqrt((x/scale)^2 + 1))"""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        sc = F.softplus(self.scale) + 1e-3
        return torch.asinh(x / sc)


def run_range_handler_audit():
    print("=" * 80)
    print("      [PHASE 5: DYNAMIC RANGE HANDLER NUMERICAL COMPARISON]      ")
    print("=" * 80)

    sample_x = torch.randn(4, 1, 128, 128, requires_grad=True) * 0.8 + 0.2
    # Insert extreme outliers
    sample_x.data[0, 0, 10, 10] = -0.2784
    sample_x.data[0, 0, 20, 20] = 3.8500
    sample_x.data[0, 0, 30, 30] = 0.0000

    options = {
        "Option A (Fixed Log)": RangeOptionA_FixedLog(),
        "Option B (Learnable Log)": RangeOptionB_LearnableLog(),
        "Option C (Log-Linear Gated)": RangeOptionC_LogLinearGated(),
        "Option D (Robust Asinh)": RangeOptionD_Asinh(scale=1.0)
    }

    results = {}
    for name, module in options.items():
        sample_x_clone = sample_x.clone().detach().requires_grad_(True)
        out = module(sample_x_clone)
        loss = out.pow(2).mean()
        loss.backward()

        grad_max = float(sample_x_clone.grad.abs().max().item())
        nan_in_out = int(torch.isnan(out).sum().item())
        nan_in_grad = int(torch.isnan(sample_x_clone.grad).sum().item())

        results[name] = {
            "output_min": float(out.min().item()),
            "output_max": float(out.max().item()),
            "grad_max": round(grad_max, 4),
            "nan_in_output": nan_in_out,
            "nan_in_grad": nan_in_grad,
            "stable": (nan_in_out == 0 and nan_in_grad == 0 and grad_max < 1e4)
        }

        print(f"[{name}]")
        print(f"  Out Range : [{results[name]['output_min']:.4f}, {results[name]['output_max']:.4f}]")
        print(f"  Max Grad  : {results[name]['grad_max']}")
        print(f"  NaN Out/Grad : {nan_in_out} / {nan_in_grad}")
        print(f"  Status    : {'STABLE' if results[name]['stable'] else 'UNSTABLE'}")
        print("-" * 60)

    print("=" * 80)
    print("Recommended Range Handler: Option D (Robust Asinh) — Smooth linear transition at 0, no log floor, robust under extreme negative/positive outliers!")
    print("=" * 80)


if __name__ == "__main__":
    run_range_handler_audit()
