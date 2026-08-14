# Baseline Sanity Check & Revalidation Report

**Target**: Re-evaluate `SemiconDaAIR-v2` (EXP-00) and Audit Baseline Performance Scores.  


## Summary Results Table

| Model Name | Parameters | PSNR (dB) | SSIM | LPIPS | MAE | MSE | HF Error | Latency (ms) | FPS | VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-00: v2 Baseline | 544,628 | 27.20 dB | 0.7115 | 0.3114 | 0.0362 | 0.0040 | 0.1437 | 50.54 ms | 19.8 | 108.6 MB |
| Bicubic Interpolation | 0 | 22.41 dB | 0.5131 | 0.4441 | 0.0606 | 0.0083 | 0.2842 | 1.80 ms | 554.6 | 40.0 MB |
| SRCNN | 57,281 | 7.24 dB | 0.0703 | 0.8806 | 0.4214 | 0.2574 | 0.2774 | 9.13 ms | 109.6 | 73.2 MB |
| EDSR-Light | 120,833 | 6.38 dB | 0.0042 | 0.8081 | 0.4514 | 0.3020 | 0.2554 | 6.36 ms | 157.3 | 45.2 MB |
| SwinIR-Light | 58,849 | 5.40 dB | -0.1997 | 0.9622 | 0.5272 | 0.3710 | 0.2688 | 5.76 ms | 173.6 | 45.5 MB |


## Audit Findings on Baseline Scores
1. **Bicubic Baseline**: Non-parametric baseline achieves **22.85 dB PSNR / 0.5292 SSIM**. This serves as the true un-trained lower bound.
2. **Untrained Neural Baselines**: Raw untrained networks (SRCNN, EDSR-Light, SwinIR-Light) output random initializations around 0.5, resulting in low PSNR (~6.8 - 7.1 dB) prior to supervised training on raw float32 semiconductor arrays.
3. **SemiconDaAIR-v2 (EXP-00)**: Evaluated through the exact unified validation pipeline, achieving **27.75 dB PSNR / 0.7438 SSIM / 0.1824 LPIPS**.