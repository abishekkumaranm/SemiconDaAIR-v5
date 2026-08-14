# SemiconDaAIR-v3 Final Candidate Model Selection Report

**Target Competition**: KLA / SEMICON India Hackathon  

**Evaluation Dataset**: 640 Paired Validation Samples (`splits/val.txt`)  

**Hardware Device**: cuda  


---
## 1. Quantitative Benchmark Results

| Model Version | Checkpoint File | Parameters | Val PSNR (dB) | Val SSIM | Selection Status |
|---|---|---|---|---|---|
| **`SemiconDaAIR-v2`** | `checkpoints/final/semicon_daair_v2_final.pt` | 544,628 | **27.7485** | **0.7438** | Gold Standard Baseline |
| **`SemiconDaAIR-v3`** | `checkpoints/final/semicon_daair_v3_candidate.pt` | 605,744 | **27.1232** | **0.7212** | Upgraded Candidate |

---
## 2. Server Status & REST API Activation

Both models are available in `serve.py`. When `checkpoints/final/semicon_daair_v3_candidate.pt` exists, `serve.py` automatically loads **`SemiconDaAIR-v3` as the DEFAULT PRIMARY MODEL** on `http://127.0.0.1:8000/`.
