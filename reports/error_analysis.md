# Error Analysis & Failure Mode Categorization (`SemiconDaAIR-v2`)

**Target**: KLA / SEMICON India Hackathon  

**Validation Split**: 640 samples (`splits/val.txt`)  


---
## 1. Executive Summary of Failure Modes

Every validation sample was analyzed across 6 metrology metrics. Failure modes were categorized into 4 primary regimes:

- **Residual Background Noise**: Low-contrast regions with speckle remnants.

- **High-Frequency Edge Loss**: Highly dense line-space gratings where Sobel error increases.

- **Extreme Dynamic Range / Negative Speckle**: Sub-zero intensity values causing local contrast compression.

- **Nominal Restoration**: High fidelity restoration (88.4% of validation set).


---
## 2. Worst 20 PSNR Validation Cases

| Rank | Filename | PSNR (dB) | SSIM | LPIPS | MAE | HF Error | Dynamic Range | Category |
|---|---|---|---|---|---|---|---|---|
| `01` | `002639.npy` | **10.52 dB** | 0.2668 | 0.8669 | 0.2380 | 0.6469 | [-0.18, 1.52] | High-Frequency Edge Loss |
| `02` | `002975.npy` | **10.53 dB** | 0.2699 | 0.8650 | 0.2372 | 0.6412 | [-0.18, 1.48] | High-Frequency Edge Loss |
| `03` | `002982.npy` | **10.56 dB** | 0.2742 | 0.8683 | 0.2370 | 0.6409 | [-0.19, 1.45] | High-Frequency Edge Loss |
| `04` | `002637.npy` | **10.58 dB** | 0.2723 | 0.8544 | 0.2367 | 0.6379 | [-0.25, 1.44] | High-Frequency Edge Loss |
| `05` | `002973.npy` | **10.58 dB** | 0.2741 | 0.8538 | 0.2362 | 0.6399 | [-0.26, 1.46] | High-Frequency Edge Loss |
| `06` | `000352.npy` | **10.65 dB** | 0.2792 | 0.8708 | 0.2340 | 0.6356 | [-0.28, 1.58] | High-Frequency Edge Loss |
| `07` | `000407.npy` | **14.99 dB** | 0.2834 | 0.8260 | 0.1416 | 0.3830 | [-0.13, 0.84] | High-Frequency Edge Loss |
| `08` | `000405.npy` | **15.83 dB** | 0.2828 | 0.8008 | 0.1287 | 0.3526 | [-0.13, 0.89] | High-Frequency Edge Loss |
| `09` | `002468.npy` | **16.93 dB** | 0.5600 | 0.4301 | 0.0992 | 0.3798 | [-0.12, 1.63] | High-Frequency Edge Loss |
| `10` | `000514.npy` | **17.83 dB** | 0.5599 | 0.2773 | 0.0906 | 0.4472 | [-0.01, 1.35] | High-Frequency Edge Loss |
| `11` | `002475.npy` | **17.96 dB** | 0.7418 | 0.1876 | 0.0809 | 0.3936 | [-0.06, 1.46] | High-Frequency Edge Loss |
| `12` | `002534.npy` | **18.11 dB** | 0.4652 | 0.5345 | 0.0972 | 0.3472 | [-0.07, 1.24] | High-Frequency Edge Loss |
| `13` | `002535.npy` | **18.12 dB** | 0.4715 | 0.5474 | 0.0974 | 0.3476 | [-0.10, 1.22] | High-Frequency Edge Loss |
| `14` | `000496.npy` | **18.29 dB** | 0.6962 | 0.2682 | 0.0777 | 0.3340 | [-0.10, 1.63] | High-Frequency Edge Loss |
| `15` | `000228.npy` | **18.30 dB** | 0.7395 | 0.1767 | 0.0814 | 0.3829 | [-0.07, 1.39] | High-Frequency Edge Loss |
| `16` | `002744.npy` | **18.51 dB** | 0.5938 | 0.2744 | 0.0894 | 0.4310 | [-0.06, 1.58] | High-Frequency Edge Loss |
| `17` | `001387.npy` | **18.68 dB** | 0.7661 | 0.0949 | 0.0821 | 0.4014 | [-0.06, 1.43] | High-Frequency Edge Loss |
| `18` | `000677.npy` | **18.85 dB** | 0.7185 | 0.1765 | 0.0807 | 0.3821 | [-0.13, 1.76] | High-Frequency Edge Loss |
| `19` | `001605.npy` | **19.26 dB** | 0.8772 | 0.1853 | 0.0836 | 0.3874 | [-0.02, 1.83] | High-Frequency Edge Loss |
| `20` | `002776.npy` | **19.28 dB** | 0.6348 | 0.3327 | 0.0684 | 0.2939 | [-0.07, 1.51] | High-Frequency Edge Loss |

---
## 3. Worst 20 SSIM Structural Cases

| Rank | Filename | SSIM | PSNR (dB) | LPIPS | HF Error | Category |
|---|---|---|---|---|---|---|
| `01` | `001977.npy` | **0.1869** | 21.66 dB | 1.0150 | 0.3338 | High-Frequency Edge Loss |
| `02` | `000399.npy` | **0.1882** | 21.26 dB | 0.8486 | 0.3110 | High-Frequency Edge Loss |
| `03` | `000398.npy` | **0.2209** | 22.26 dB | 0.7980 | 0.2665 | High-Frequency Edge Loss |
| `04` | `003070.npy` | **0.2384** | 23.19 dB | 0.8818 | 0.2413 | Residual Background Noise |
| `05` | `001208.npy` | **0.2506** | 20.60 dB | 0.8067 | 0.3354 | High-Frequency Edge Loss |
| `06` | `003071.npy` | **0.2506** | 23.50 dB | 0.8507 | 0.2353 | Residual Background Noise |
| `07` | `002607.npy` | **0.2519** | 23.08 dB | 0.8187 | 0.1906 | Residual Background Noise |
| `08` | `002694.npy` | **0.2530** | 22.68 dB | 0.9210 | 0.2667 | High-Frequency Edge Loss |
| `09` | `000642.npy` | **0.2619** | 20.22 dB | 0.9251 | 0.4108 | High-Frequency Edge Loss |
| `10` | `002639.npy` | **0.2668** | 10.52 dB | 0.8669 | 0.6469 | High-Frequency Edge Loss |
| `11` | `002975.npy` | **0.2699** | 10.53 dB | 0.8650 | 0.6412 | High-Frequency Edge Loss |
| `12` | `002517.npy` | **0.2702** | 23.53 dB | 0.8804 | 0.2415 | Residual Background Noise |
| `13` | `002637.npy` | **0.2723** | 10.58 dB | 0.8544 | 0.6379 | High-Frequency Edge Loss |
| `14` | `002713.npy` | **0.2724** | 20.73 dB | 0.8113 | 0.2890 | High-Frequency Edge Loss |
| `15` | `002714.npy` | **0.2734** | 21.09 dB | 0.8425 | 0.2876 | High-Frequency Edge Loss |
| `16` | `002973.npy` | **0.2741** | 10.58 dB | 0.8538 | 0.6399 | High-Frequency Edge Loss |
| `17` | `002982.npy` | **0.2742** | 10.56 dB | 0.8683 | 0.6409 | High-Frequency Edge Loss |
| `18` | `003190.npy` | **0.2758** | 23.23 dB | 0.8775 | 0.2573 | High-Frequency Edge Loss |
| `19` | `000352.npy` | **0.2792** | 10.65 dB | 0.8708 | 0.6356 | High-Frequency Edge Loss |
| `20` | `000405.npy` | **0.2828** | 15.83 dB | 0.8008 | 0.3526 | High-Frequency Edge Loss |

---
## 4. Highest High-Frequency (HF) Sobel Gradient Error Cases

| Rank | Filename | HF Error | PSNR (dB) | SSIM | LPIPS | Category |
|---|---|---|---|---|---|---|
| `01` | `002639.npy` | **0.6469** | 10.52 dB | 0.2668 | 0.8669 | High-Frequency Edge Loss |
| `02` | `002975.npy` | **0.6412** | 10.53 dB | 0.2699 | 0.8650 | High-Frequency Edge Loss |
| `03` | `002982.npy` | **0.6409** | 10.56 dB | 0.2742 | 0.8683 | High-Frequency Edge Loss |
| `04` | `002973.npy` | **0.6399** | 10.58 dB | 0.2741 | 0.8538 | High-Frequency Edge Loss |
| `05` | `002637.npy` | **0.6379** | 10.58 dB | 0.2723 | 0.8544 | High-Frequency Edge Loss |
| `06` | `000352.npy` | **0.6356** | 10.65 dB | 0.2792 | 0.8708 | High-Frequency Edge Loss |
| `07` | `000514.npy` | **0.4472** | 17.83 dB | 0.5599 | 0.2773 | High-Frequency Edge Loss |
| `08` | `002744.npy` | **0.4310** | 18.51 dB | 0.5938 | 0.2744 | High-Frequency Edge Loss |
| `09` | `000642.npy` | **0.4108** | 20.22 dB | 0.2619 | 0.9251 | High-Frequency Edge Loss |
| `10` | `001387.npy` | **0.4014** | 18.68 dB | 0.7661 | 0.0949 | High-Frequency Edge Loss |
| `11` | `002475.npy` | **0.3936** | 17.96 dB | 0.7418 | 0.1876 | High-Frequency Edge Loss |
| `12` | `001605.npy` | **0.3874** | 19.26 dB | 0.8772 | 0.1853 | High-Frequency Edge Loss |
| `13` | `000407.npy` | **0.3830** | 14.99 dB | 0.2834 | 0.8260 | High-Frequency Edge Loss |
| `14` | `000228.npy` | **0.3829** | 18.30 dB | 0.7395 | 0.1767 | High-Frequency Edge Loss |
| `15` | `000677.npy` | **0.3821** | 18.85 dB | 0.7185 | 0.1765 | High-Frequency Edge Loss |
| `16` | `002468.npy` | **0.3798** | 16.93 dB | 0.5600 | 0.4301 | High-Frequency Edge Loss |
| `17` | `002124.npy` | **0.3645** | 19.75 dB | 0.6909 | 0.2088 | High-Frequency Edge Loss |
| `18` | `002127.npy` | **0.3594** | 19.50 dB | 0.7031 | 0.1697 | High-Frequency Edge Loss |
| `19` | `000405.npy` | **0.3526** | 15.83 dB | 0.2828 | 0.8008 | High-Frequency Edge Loss |
| `20` | `002535.npy` | **0.3476** | 18.12 dB | 0.4715 | 0.5474 | High-Frequency Edge Loss |