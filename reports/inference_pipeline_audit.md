# Inference Pipeline Audit Report (`serve.py` & Web Dashboard)

**Target Endpoint**: `POST /api/restore`  


---
## 1. End-to-End Pipeline Verification

1. **Upload & Format Handling**: Supports `.png`, `.jpg`, `.tif`, `.tiff`, `.npy`.

2. **Dynamic Range Preservation**: Preserves signed float32 values ($[-0.2786, 2.1580]$); no silent clipping to [0,1].

3. **PyTorch Model Execution**: Singleton `SemiconDaAIR-v2` loaded once on GPU using `torch.inference_mode()`.

4. **Tiled Overlapping Patch Inference**: Full-resolution images (e.g. `image.png` 533x684) are tiled into 128x128 patches with stride 64, super-resolved to 256x256 per patch, and blended cleanly into $1368 	imes 1066$ output.

5. **Raw vs Display Representation**: Raw float32 numpy array saved as downloadable `.npy`; visualization-only PNG generated for browser canvas.
