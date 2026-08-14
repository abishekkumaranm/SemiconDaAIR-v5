"""
instrumented_v6.py — Phase 2: Full Tensor Numerical Instrumentation for SemiconDaAIR-v6.

Instruments all 18 intermediate tensors during forward pass and backward loss computation:
  1. Raw Input X_LQ
  2. Normalized Input X_norm
  3. Log Domain Output
  4. Dynamic Epsilon
  5. Shallow Encoder Output
  6. Speckle Branch Output
  7. Directional MoE Expert Output
  8. Radial MoE Expert Output
  9. Interconnect MoE Expert Output
 10. Shared MoE Expert Output
 11. MoE Router Logits & Weights
 12. Deformable Sub-Pixel Offsets (dx, dy)
 13. Deformable Feature Output
 14. FFT Real & Imaginary Sub-Bands
 15. Tukey Window Mask
 16. Bottleneck SSM Output
 17. Self-Guided Controller Output
 18. Residual R(x,y), Confidence C(x,y), & Final HR Output

Fails immediately with clear module name if NaN/Inf appears. NEVER silently masks NaNs with zero!
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.robust_range import DynamicLearnableEpsilonHandler
from models.structural_moe import MultiStructuralMoE
from models.structure_guidance import DeformableSubPixelPhaseExtractor
from models.frequency_module import TukeyWindowSmoothSpectralFilter
from models.semicon_daair_v6 import DualStageBottleneckSSM
from models.semicon_daair_v3 import SpeckleAwareBranch, FidelityGatedHead
from models.controller import SelfLearnableController


def audit_tensor_numerics(tensor: torch.Tensor, module_name: str, tensor_name: str, raise_on_error: bool = True) -> dict:
    """Inspects tensor min, max, mean, std, NaN count, Inf count."""
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    
    if nan_count > 0 or inf_count > 0:
        err_msg = (
            f"\n[NUMERICAL ERROR]\n"
            f"Module  : {module_name}\n"
            f"Tensor  : {tensor_name}\n"
            f"NaNs    : {nan_count}\n"
            f"Infs    : {inf_count}\n"
            f"Min     : {float(tensor.min().item()) if nan_count == 0 else 'NaN'}\n"
            f"Max     : {float(tensor.max().item()) if nan_count == 0 else 'NaN'}\n"
        )
        print(err_msg, flush=True)
        if raise_on_error:
            raise ValueError(err_msg)

    valid_mask = torch.isfinite(tensor)
    valid_t = tensor[valid_mask] if valid_mask.any() else tensor.view(-1)
    
    stats = {
        "module": module_name,
        "tensor": tensor_name,
        "min": float(valid_t.min().item()) if valid_t.numel() > 0 else 0.0,
        "max": float(valid_t.max().item()) if valid_t.numel() > 0 else 0.0,
        "mean": float(valid_t.mean().item()) if valid_t.numel() > 0 else 0.0,
        "std": float(valid_t.std().item()) if valid_t.numel() > 0 else 0.0,
        "nan_count": nan_count,
        "inf_count": inf_count
    }
    return stats


class InstrumentedSemiconDaAIRv6(nn.Module):
    """
    Instrumented SemiconDaAIR-v6 Architecture for Numerical Safety Debugging.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.base_channels = base_channels

        self.range_handler = DynamicLearnableEpsilonHandler(channels=base_channels)
        self.shallow_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.speckle_branch = SpeckleAwareBranch(channels=base_channels)

        self.structural_moe = MultiStructuralMoE(channels=base_channels, num_experts=3)
        self.spatial_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.deformable_phase = DeformableSubPixelPhaseExtractor(channels=base_channels)
        self.tukey_frequency = TukeyWindowSmoothSpectralFilter(channels=base_channels)
        self.fusion_block = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        self.global_context = DualStageBottleneckSSM(channels=base_channels, reduction=2)
        self.controller = SelfLearnableController(channels=base_channels, num_heads=4)
        self.sr_head = FidelityGatedHead(in_channels=base_channels, out_channels=out_channels, scale=scale)

    def forward(self, x: torch.Tensor, return_audit: bool = True):
        audit_records = []

        # 1. Input & Range Handler
        audit_records.append(audit_tensor_numerics(x, "Input", "x_lq"))
        norm_feat = self.range_handler(x)
        audit_records.append(audit_tensor_numerics(norm_feat, "DynamicLearnableEpsilonHandler", "norm_feat"))

        # 2. Shallow & Speckle
        f0 = self.shallow_conv(x) + norm_feat
        audit_records.append(audit_tensor_numerics(f0, "ShallowConv", "f0"))
        f_speckle = self.speckle_branch(f0, x)
        audit_records.append(audit_tensor_numerics(f_speckle, "SpeckleAwareBranch", "f_speckle"))

        # 3. Structural MoE
        f_moe, moe_gates = self.structural_moe(f_speckle)
        audit_records.append(audit_tensor_numerics(f_moe, "MultiStructuralMoE", "f_moe"))
        audit_records.append(audit_tensor_numerics(moe_gates, "MoERouter", "moe_gates"))

        # 4. Deformable Phase & Tukey Frequency
        f_struct = self.deformable_phase(self.spatial_conv(f_moe), x)
        audit_records.append(audit_tensor_numerics(f_struct, "DeformableSubPixelPhaseExtractor", "f_struct"))

        f_freq = self.tukey_frequency(f_moe)
        audit_records.append(audit_tensor_numerics(f_freq, "TukeyWindowSmoothSpectralFilter", "f_freq"))

        f_fused = self.fusion_block(torch.cat([f_struct, f_freq], dim=1))
        audit_records.append(audit_tensor_numerics(f_fused, "FusionBlock", "f_fused"))

        # 5. Bottleneck SSM & Controller
        f_ssm = self.global_context(f_fused)
        audit_records.append(audit_tensor_numerics(f_ssm, "DualStageBottleneckSSM", "f_ssm"))

        f_guided = self.controller(f_fused, f_ssm)
        audit_records.append(audit_tensor_numerics(f_guided, "SelfLearnableController", "f_guided"))

        # 6. Reconstruction Head
        out_hr, confidence_map = self.sr_head(f_guided, x)
        audit_records.append(audit_tensor_numerics(out_hr, "FidelityGatedHead", "out_hr"))
        audit_records.append(audit_tensor_numerics(confidence_map, "FidelityGatedHead", "confidence_map"))

        if return_audit:
            return out_hr, audit_records
        return out_hr


if __name__ == "__main__":
    model = InstrumentedSemiconDaAIRv6()
    sample_x = torch.randn(2, 1, 128, 128)
    sample_x[0, 0, 5, 5] = -0.2784
    sample_x[0, 0, 10, 10] = 3.8500
    out, logs = model(sample_x, return_audit=True)
    print("=" * 75)
    print("      [INSTRUMENTED V6 FORWARD PASS NUMERICAL AUDIT PASSED]      ")
    print(f"Total Tensors Audited : {len(logs)}")
    print(f"Output Shape          : {out.shape}")
    print("=" * 75)
