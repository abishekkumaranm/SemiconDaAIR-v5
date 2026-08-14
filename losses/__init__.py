"""
losses package initialization.
"""

from .pixel_loss import CharbonnierLoss
from .structure_loss import SobelGradientEdgeLoss, LaplacianStructureLoss
from .frequency_loss import FourierFrequencyLoss
from .total_loss import SemiconDaAIRv6CompositeLoss

__all__ = [
    "CharbonnierLoss",
    "SobelGradientEdgeLoss",
    "LaplacianStructureLoss",
    "FourierFrequencyLoss",
    "SemiconDaAIRv6CompositeLoss"
]
