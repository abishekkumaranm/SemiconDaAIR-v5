# SemiconDaAIR-v3 Controlled Ablation Report

**Target Project**: KLA / SEMICON India Challenge  

**Evaluation Split**: 640 Paired Validation Samples (`splits/val.txt`)  

**Hardware Device**: cuda  


---
## 1. Quantitative Ablation Benchmark Table

| Experiment ID | Architecture Description | Parameters | Val PSNR (dB) | Val SSIM | Val MAE | Fidelity Risk | OOD PSNR | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|---|
| **EXP-A** | `SemiconDaAIR-v2` (Gold Standard Baseline) | 544,628 | **27.6705** | **0.7425** | 0.0332 | 2.8755 | 18.9647 | 36.71 ms | 27.2 FPS |
| **EXP-B** | `SemiconDaAIR-v2` + `FidelityGatedHead` | 607,349 | **26.8422** | **0.7121** | 0.0361 | 3.0886 | 18.9779 | 27.28 ms | 36.7 FPS |

---
## 2. Performance Delta vs Baseline v2

- **PSNR Delta**: `-0.8283 dB`

- **SSIM Delta**: `-0.0304`

- **Fidelity Risk Delta**: `+0.2131` (Lower is better)

- **Latency Delta**: `-9.43 ms`


---
## 3. Automatic Model Selection Decision

**Decision Status**: `PRESERVED: SemiconDaAIR-v2 baseline remains production champion!`
