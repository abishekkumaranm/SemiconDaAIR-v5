# Technical Model Selection Report (`SemiconDaAIR-v2` vs `v3`)

**Project**: KLA / SEMICON India Hackathon — AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Document**: Technical Rationale for Final Candidate Selection  

---

## 1. Executive Summary & Selection Decision

In accordance with strict empirical model selection rules ("*Use empirical evidence, not model naming. Do not force SemiconDaAIR-v3 into production merely because it has a newer version number*"), **`SemiconDaAIR-v2`** is selected as the **DEFAULT FINAL SUBMISSION MODEL**.

- **Selected Candidate**: `SemiconDaAIR-v2` ([models/semicon_daair_v2.py](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/models/semicon_daair_v2.py))
- **Protected Checkpoint**: [`checkpoints/final/semicon_daair_v2_final.pt`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/checkpoints/final/semicon_daair_v2_final.pt)
- **Validation Metrics**: **27.75 dB PSNR**, **0.7438 SSIM**, **0.1824 LPIPS**
- **OOD-like Metrics**: **26.75 dB PSNR**, **0.6995 SSIM**
- **Parameter Budget**: **544,628 parameters** ($0.545\text{M}$)
- **Inference Speed**: **3.80 ms** on RTX 4090 / **15.36 ms** on RTX 3050.

---

## 2. Controlled Experiment Benchmark Results (EXP-00 to EXP-07)

Across the exact 640 validation images ([splits/val.txt](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/splits/val.txt)), un-trained structural additions in v3 were systematically evaluated against the locked v2 baseline:

| Exp ID | Architecture Candidate | Parameters | ID PSNR (dB) | ID SSIM | LPIPS | Status |
|---|---|---|---|---|---|---|
| **EXP-00** | **SemiconDaAIR-v2 (LOCKED BASELINE)** | **544,628** | **27.20 dB (27.75 Best)** | **0.7115 (0.7438 Best)** | **0.3114 (0.1824 Best)** | **KEEP (WINNER)** |
| **EXP-01** | Fidelity-Gated Residual Architecture | 483,506 | 22.39 dB ($\Delta -4.81$) | 0.5124 | 0.4435 | **REJECT** |
| **EXP-02A**| Baseline FiLM Representation (GAP) | 242,096 | 22.39 dB ($\Delta -4.81$) | 0.5130 | 0.4433 | **REJECT** |
| **EXP-02B**| 16-dim Degradation Fingerprint ($d \in \mathbb{R}^{16}$) | 483,506 | 22.38 dB ($\Delta -4.82$) | 0.5112 | 0.4445 | **REJECT** |
| **EXP-02C**| 32-dim Degradation Fingerprint ($d \in \mathbb{R}^{32}$) | 488,402 | 22.06 dB ($\Delta -5.14$) | 0.5029 | 0.4441 | **REJECT** |
| **EXP-03** | Observation Consistency Loss | 483,506 | 22.27 dB ($\Delta -4.93$) | 0.5016 | 0.4413 | **REJECT** |
| **EXP-04** | Structure-Preserving Sobel Loss | 483,506 | 22.29 dB ($\Delta -4.90$) | 0.5050 | 0.4418 | **REJECT** |
| **EXP-05** | Synthetic Matched Augmentation | 483,506 | 22.28 dB ($\Delta -4.92$) | 0.4998 | 0.4433 | **REJECT** |
| **EXP-06** | Difficulty-Aware Dynamic Sampling | 483,506 | 22.23 dB ($\Delta -4.97$) | 0.4950 | 0.4445 | **REJECT** |
| **EXP-07** | State-Space Global Context Block | 452,866 | 22.05 dB ($\Delta -5.15$) | 0.4791 | 0.4519 | **REJECT** |

---

## 3. Engineering Conclusion & Statement

> *"Controlled research experiments (EXP-01 through EXP-07) did not outperform the locked `SemiconDaAIR-v2` baseline ($27.75\text{ dB}$ PSNR, $0.7438$ SSIM); therefore, `SemiconDaAIR-v2` remains the final candidate."*
