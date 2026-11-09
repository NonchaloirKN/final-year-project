"""
Loss functions for the FCN05 multimodal registration pipeline.
"""

from src.losses.contrastive_loss import MultiPositiveInfoNCELoss
from src.losses.pose_loss import MultiModalPoseLoss, chordal_frobenius_loss

__all__ = [
    "MultiPositiveInfoNCELoss",
    "MultiModalPoseLoss",
    "chordal_frobenius_loss",
]
