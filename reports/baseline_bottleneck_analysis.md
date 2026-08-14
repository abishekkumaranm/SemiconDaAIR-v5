# SemiconDaAIR-v2 Baseline Bottleneck Analysis Report

**Target Architecture**: `SemiconDaAIR-v2` ($544,628$ parameters)  

**Baseline Benchmark**: **27.75 dB PSNR**, **0.7438 SSIM**, **0.0335 MAE**  


---
## 1. Identified Architectural Bottlenecks

1. **Background Speckle Remnants**: Multiplicative e-beam speckle noise leaves minor low-frequency grain in uniform silicon substrate regions.

   - *Proposed Solution*: Add 3-layer DnCNN-inspired residual denoising block (EXP-01).


2. **Line-Edge Gradient Softening**: High-density line-space gratings show minor edge softening under 2x super-resolution.

   - *Proposed Solution*: Add RCAN-style Residual Channel Attention Block (RCAB) in feature fusion (EXP-03).


3. **Spatial Resolution Scale Mismatch on Full Images**: Direct un-tiled forward pass on large images ($> 256 	imes 256$) distorts neural filter receptive fields.

   - *Verified Solution*: Overlapping patch-based tiled inference (`tiled_inference`) with 2D Hanning window blending in `serve.py`.
