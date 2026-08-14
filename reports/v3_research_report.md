# SemiconDaAIR-v3 Comprehensive Research & Experimental Report

**Target**: Discover the strongest lightweight, faithful, and generalizable semiconductor image restoration model.  

**Baseline Reference (EXP-00)**: `SemiconDaAIR-v2` (PSNR = 27.75 dB, SSIM = 0.7438, Params = 544,628).  


## Master Controlled Experiments Results Table

| Experiment | Description | Params | ID PSNR (dB) | ID SSIM | LPIPS | MAE | HF Error | OOD PSNR | OOD SSIM | Latency (ms) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `EXP-00` | LOCKED v2 BASELINE (v2 best_psnr.pt) | 544,628 | 27.20 dB | 0.7115 | 0.3114 | 0.0362 | 0.1437 | 26.75 dB | 0.6995 | 35.76 ms | **KEEP** |
| `EXP-01` | Fidelity-Gated Residual (base + confidence * residual) | 483,506 | 22.39 dB | 0.5124 | 0.4435 | 0.0606 | 0.2841 | 21.69 dB | 0.4866 | 13.47 ms | **REJECT** |
| `EXP-02A` | Baseline FiLM Representation (GAP) | 242,096 | 22.39 dB | 0.5130 | 0.4433 | 0.0606 | 0.2843 | 21.67 dB | 0.4868 | 10.03 ms | **REJECT** |
| `EXP-02B` | 16-dim Unlabeled Degradation Fingerprint (d in R^16) | 483,506 | 22.38 dB | 0.5112 | 0.4445 | 0.0607 | 0.2842 | 21.67 dB | 0.4854 | 13.25 ms | **REJECT** |
| `EXP-02C` | 32-dim Unlabeled Degradation Fingerprint (d in R^32) | 488,402 | 22.06 dB | 0.5029 | 0.4441 | 0.0632 | 0.2845 | 21.40 dB | 0.4771 | 20.00 ms | **REJECT** |
| `EXP-03` | Differentiable Forward Degradation Observation Consistency | 483,506 | 22.27 dB | 0.5016 | 0.4413 | 0.0616 | 0.2844 | 21.57 dB | 0.4762 | 20.90 ms | **REJECT** |
| `EXP-04` | Structure-Preserving Loss (Sobel/Laplacian) | 483,506 | 22.29 dB | 0.5050 | 0.4418 | 0.0613 | 0.2845 | 21.60 dB | 0.4792 | 12.92 ms | **REJECT** |
| `EXP-05` | Realistic Synthetic Matched Degradation Augmentation | 483,506 | 22.28 dB | 0.4998 | 0.4433 | 0.0619 | 0.2843 | 21.59 dB | 0.4739 | 13.15 ms | **REJECT** |
| `EXP-06` | Difficulty-Aware Dynamic Sampling | 483,506 | 22.23 dB | 0.4950 | 0.4445 | 0.0625 | 0.2843 | 21.53 dB | 0.4689 | 13.20 ms | **REJECT** |
| `EXP-07` | Lightweight State-Space Global Context Block | 452,866 | 22.05 dB | 0.4791 | 0.4519 | 0.0636 | 0.2838 | 21.39 dB | 0.4559 | 13.33 ms | **REJECT** |