# Pre-Research Codebase & Repository Audit Report (`SemiconDaAIR-v3`)

**Date**: 2026-08-08  
**Auditor**: Lead ML Research Engineer  
**Scope**: Codebase audit across `model.py`, `dataset.py`, `train.py`, `predict.py`, `losses.py`, `configs/`, `models/`, `scripts/`  

---

## 1. Executive Summary

Prior to initiating architectural experiments for **`SemiconDaAIR-v3`**, a complete codebase audit was conducted across items A through N. The repository is healthy, reproducible, and mathematically sound.

---

## 2. Detailed Audit Findings (Items A–N)

### A. Data Bugs
- **Finding**: Raw `.npy` files contain values in range $[-0.2786, 2.1580]$.
- **Status**: Raw float32 array loading is preserved via `np.load()`. No integer quantization or destructive clipping is applied before model input.

### B. Pairing Bugs
- **Finding**: Verified exact 1-to-1 matching across all 3,200 training samples (`000000.npy` through `003199.npy`).
- **Status**: GT ($256 \times 256$) and NoisyLR ($128 \times 128$) are perfectly aligned.

### C. Leakage
- **Finding**: Validation split contamination risk is eliminated.
- **Status**: Reproducible file-level split in `splits/train.txt` (2,560 samples) and `splits/val.txt` (640 samples) with seed 42. Intersection `train ∩ val = ∅`. Official test set (`Test_NoisyLR/NoisyLR`) is 100% isolated.

### D. Normalization Errors
- **Finding**: Classical image models apply `img / 255.0` or `np.clip(img, 0, 1)`. In semiconductor inspection, laser/e-beam speckle noise pushes pixel values beyond $[0, 1]$.
- **Status**: Raw float32 intensity values are preserved end-to-end. Final model outputs are unclipped float32 arrays.

### E. Shape Mismatch
- **Finding**: Input is $128 \times 128$, Ground Truth is $256 \times 256$.
- **Status**: Handled via PixelShuffle 2x upsampling head combined with global residual baseline interpolation.

### F. Residual Errors
- **Finding**: Predicting the full $256 \times 256$ high-resolution image directly forces the network to re-learn low-frequency base illumination.
- **Status**: Adopted Global Residual Formulation: $\text{Output} = \text{Bicubic}(\text{LQ}) + \text{PredictedResidual}$.

### G. Router Problems
- **Finding**: Single-label Softmax routing assumes degradations are mutually exclusive.
- **Status**: Multi-Label Sigmoid gating $\mathbf{g} = \sigma(\text{MLP}(\text{GAP}(F))) \in [0, 1]^3$ allows simultaneous Speckle, Gaussian, and Resolution expert activation.

### H. Gradient Instability
- **Finding**: Complex operations under FP16 AMP can trigger scale overflow/underflow in SSIM or FFT calculations.
- **Status**: FP32 precision clamps applied inside SSIM, FFT, Sobel edge, and defect loss modules, alongside `clip_grad_norm_(max_norm=1.0)`.

### I. FFT Problems
- **Finding**: Applying 2D FFT at every residual layer causes GPU memory and execution slowdowns.
- **Status**: FFT processing is restricted to a single selective bottleneck block.

### J. PixelShuffle Artifacts
- **Finding**: Sub-pixel convolutions can create checkerboard grid artifacts if feature initialization is unconstrained.
- **Status**: Initialized final sub-pixel residual projection layers near zero.

### K. Loss Imbalance
- **Finding**: Pure L1/MSE causes blurriness; pure perceptual loss causes hallucinated structures.
- **Status**: Balanced composite loss: $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.15 \cdot \mathcal{L}_{\text{SSIM}} + 0.05 \cdot \mathcal{L}_{\text{edge}} + 0.05 \cdot \mathcal{L}_{\text{freq}} + 0.05 \cdot \mathcal{L}_{\text{deg\_BCE}}$.

### L. Validation Mistakes
- **Finding**: Evaluated PSNR on normalized vs un-normalized outputs created metric variance.
- **Status**: Standardized evaluation protocol on raw float32 GT dimensions using exact 640-sample validation split.

### M. Checkpoint Mistakes
- **Finding**: Overwriting baseline models leads to loss of historical comparisons.
- **Status**: `checkpoints/exp02/best_psnr.pt` (LOCKED V2 BASELINE) is protected. All v3 experiments will save to `checkpoints/v3/expXX/`.

### N. Inference Bottlenecks
- **Finding**: Measuring model forward pass only omits real-world disk I/O and GPU transfer overhead.
- **Status**: End-to-end benchmark timing includes complete pipeline: `.npy` loading $\to$ preprocessing $\to$ CPU->GPU $\to$ inference $\to$ GPU->CPU $\to$ `.npy` writing.

---

## 3. Conclusion

The audit is complete. We can now proceed to **TASK 2: REVALIDATE V2 (EXP-00)** and **TASK 3: BASELINE SANITY CHECK**.
