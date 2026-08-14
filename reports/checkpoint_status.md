# 🛡️ Checkpoint Integrity Validation Report

- **Checkpoint File**: `checkpoints/v5_backup/semicon_daair_v5_candidate.pt`
- **Validation Status**: `VALIDATED`
- **File Size**: `2,264,690 bytes` (2.16 MB)
- **Model Architecture**: `SemiconDaAIRv5`
- **Total Parameter Count**: `555,141`

## 🧪 Detailed Validation Tests
1. **Checkpoint File Exists**: `PASS`
2. **PyTorch `torch.load` Ingestion**: `PASS`
3. **State Dict Key Match**: `PASS` (0 missing, 0 unexpected)
4. **Finite Parameter Weights**: `True` (0 NaNs, 0 Infs)
5. **Dummy Inference Test**: `PASS` (`[1, 1, 128, 128]` $\to$ `[1, 1, 256, 256]`)
6. **Output Tensor Value Fiteness**: `True` (0 NaNs, 0 Infs)
7. **Spatial Resolution Expansion**: `2x PixelShuffle Validated`

---
*Generated automatically by `tools/validate_checkpoint.py`*
