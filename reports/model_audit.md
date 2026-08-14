# SemiconDaAIR Architecture & Codebase Comprehensive Audit Report

**Date**: 2026-08-08  
**Target Competition**: KLA / SEMICON India — AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Auditor**: Lead ML Engineer  

---

## Executive Summary

A complete line-by-line audit of the dataset, data loading, normalization, model architecture, loss functions, validation framework, and inference pipeline was conducted. This document details all identified technical bottlenecks, risks, and findings (Categories A–N) prior to architectural evolution into `SemiconDaAIR-v2`.

---

## Detailed Audit Findings (Categories A–N)

### A. Data Bugs
- **Finding**: Raw `.npy` files contain out-of-range float32 values (min: $-0.2786$, max: $2.1580$).
- **Status**: Raw float32 array loading preserved via `np.load()`. No integer quantizations or premature clamping to $[0, 1]$ applied before model input.

### B. Pairing Bugs
- **Finding**: Verified exact 1-to-1 matching across all 3,200 training samples (`000000.npy` to `003199.npy`).
- **Status**: GT ($256 \times 256$) and NoisyLR ($128 \times 128$) are perfectly aligned.

### C. Normalization Bugs
- **Finding**: Classical image restoration models apply `img / 255.0` or `np.clip(img, 0, 1)`. In semiconductor inspection, laser/e-beam speckle noise pushes pixel values beyond $[0, 1]$.
- **Action**: Input data is fed in raw float32 format. Normalization is strictly non-destructive.

### D. Leakage
- **Finding**: Previously, random validation patch splitting risked crop contamination.
- **Action**: Created reproducible file-level 80/20 split (`splits/train.txt`: 2,560 samples, `splits/val.txt`: 640 samples) with seed 42. Test set (400 samples) is strictly isolated.

### E. Shape Mismatch
- **Finding**: Input is $128 \times 128$, Ground Truth is $256 \times 256$.
- **Status**: Handled via PixelShuffle 2x upsampling head combined with global residual baseline interpolation.

### F. Incorrect Residual Formulation
- **Finding**: Directly predicting the full $256 \times 256$ high-resolution image forces the network to learn low-frequency base illumination from scratch.
- **Action**: Adopted Global Residual Formulation: $\text{Output} = \text{Bicubic}(\text{LQ}) + \text{PredictedResidual}$.

### G. Router Problems
- **Finding**: Original paper used Top-1 Softmax routing. In semiconductor inspection, Speckle, Gaussian, and Resolution degradations coexist simultaneously.
- **Action**: Router uses Multi-Label Sigmoid gating: $\mathbf{g} = \sigma(\text{MLP}(\text{GAP}(F))) \in [0, 1]^3$.

### H. Gradient Instability
- **Finding**: Complex operations in FP16 AMP mode could trigger underflow/overflow.
- **Action**: Added `x.to(torch.float32)` inside 2D FFT sub-band modules and used `torch.amp.GradScaler`.

### I. FFT Implementation Problems
- **Finding**: Early versions attempted full 2D FFT operations at every residual layer, causing GPU memory and execution slowdowns.
- **Action**: FFT frequency processing is restricted to a single selective bottleneck block.

### J. PixelShuffle Artifacts
- **Finding**: Sub-pixel convolutions can create high-frequency checkerboard grid patterns if feature initialization is unconstrained.
- **Action**: Initialized final sub-pixel residual projection layers near zero with `PReLU` activation.

### K. Loss Imbalance
- **Finding**: Pure L1/MSE losses cause blurriness; pure perceptual losses cause hallucinated non-existent semiconductor structures.
- **Action**: Formulated composite loss: $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.15 \cdot \mathcal{L}_{\text{SSIM}} + 0.05 \cdot \mathcal{L}_{\text{edge}} + 0.05 \cdot \mathcal{L}_{\text{frequency}}$.

### L. Validation Mistakes
- **Finding**: Evaluating PSNR on normalized vs un-normalized outputs created metric variance.
- **Action**: Standardized evaluation protocol on raw float32 Ground Truth dimensions using exact 640-sample validation split.

### M. Checkpoint-Selection Mistakes
- **Finding**: Selecting checkpoints based on training loss led to mild overfitting.
- **Action**: Implemented early stopping and checkpoint selection strictly based on validation PSNR/SSIM scores.

### N. Inference Bottlenecks
- **Finding**: Disk I/O, CPU-to-GPU transfer, and single-image saving added latency.
- **Action**: End-to-end benchmark timing now includes complete I/O: `.npy` loading $\to$ GPU transfer $\to$ forward pass $\to$ CPU transfer $\to$ `.npy` writing.

---

## Conclusion & Architecture Roadmap

With the audit complete and the leakage-free validation split established (`splits/train.txt` & `splits/val.txt`), we proceed to building competitor baselines and evolving the model into **`SemiconDaAIR-v2`**.
