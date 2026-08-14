# 🔍 Repository Forensic Audit Report

**Project**: `SemiconDaAIR`  
**Audit Target**: KLA / SEMICON India Hackathon 2026 Codebase  
**Execution Timestamp**: 2026-08-12  

---

## 📌 1. Found Files & Directory Inventory

| File / Component | Path | File Type / Format | Status / Size |
| :--- | :--- | :--- | :---: |
| **Model Architecture Class** | [`models/semicon_daair_v5.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/models/semicon_daair_v5.py) | Python source (`SemiconDaAIRv5`) | **FOUND** |
| **Protected Champion Checkpoint** | [`checkpoints/v5_backup/semicon_daair_v5_candidate.pt`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/checkpoints/v5_backup/semicon_daair_v5_candidate.pt) | PyTorch State Dict (`.pt`) | **FOUND** (2.22 MB) |
| **Robust Range Handler** | [`models/robust_range.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/models/robust_range.py) | Python source (`RobustAsinhRangeHandler`) | **FOUND** |
| **Spectral Frequency Module** | [`models/frequency_module.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/models/frequency_module.py) | Python source (`TukeyWindowSmoothSpectralFilter`) | **FOUND** |
| **Reconstruction Head** | [`models/semicon_daair_v3.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/models/semicon_daair_v3.py) | Python source (`FidelityGatedHead`) | **FOUND** |
| **Evaluation CLI Script** | [`evaluate.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/evaluate.py) | Python standalone script | **FOUND** |
| **Live Server Engine** | [`serve.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/serve.py) | Python HTTP web server | **FOUND** |
| **Training Pipeline** | [`training/train_v6_staged.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/training/train_v6_staged.py) | PyTorch staged training script | **FOUND** |
| **Dataset Loader** | [`dataset.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/dataset.py) | PyTorch Dataset class | **FOUND** |
| **Metrics Suite** | [`evaluation/metrics.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/evaluation/metrics.py) | Metric evaluation functions | **FOUND** |

---

## 🔬 2. Actual Measured Model & Checkpoint Attributes

1. **Actual Model Class**: `SemiconDaAIRv5` in [`models/semicon_daair_v5.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/models/semicon_daair_v5.py)
2. **Actual Parameter Count**: **`555,141 parameters`** (measured directly via `sum(p.numel() for p in model.parameters())`).
3. **Actual Checkpoint Path**: [`checkpoints/v5_backup/semicon_daair_v5_candidate.pt`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/checkpoints/v5_backup/semicon_daair_v5_candidate.pt)
4. **Actual Checkpoint Size**: `2,224,931 bytes` ($2.22\text{ MB}$).
5. **Actual Input Shape**: `[B, 1, H, W]` (Single-channel float32 array, signed range `[-0.2786, 2.1580]`).
6. **Actual Output Shape**: `[B, 1, 2*H, 2*W]` ($2\times$ PixelShuffle Super-Resolution spatial expansion).
7. **Actual Normalization**: Inverse Hyperbolic Sine $\text{asinh}(X / s)$ with learnable softplus scaling parameter $s$.
8. **Actual Baseline Metrics**: **`28.0340 dB PSNR`**, **`0.7448 SSIM`**, **`0.0325 MAE`**, **`0.3129 LPIPS`**.

---

## ⚡ 3. Audit Summary & Conclusion

- **Checkpoint Availability**: **`VALIDATED`** ([`checkpoints/v5_backup/semicon_daair_v5_candidate.pt`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/checkpoints/v5_backup/semicon_daair_v5_candidate.pt) is fully intact and verified).
- **Architecture Compatibility**: **`100% MATCH`** (State dict loads cleanly with 0 missing keys and 0 unexpected keys).
- **Documentation Alignment**: Measured parameter count (`555,141`) matches documentation claims exactly.
