"""
DualEncoderResNet18: Dual-Encoder Architecture for Cross-Modal SVR Pre-Training.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Implements dual parallel 2D ResNet-18 feature extraction backbones for 1-channel
medical slices with weight-sharing (Siamese mode) toggle and linear projection heads.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def _build_resnet18_backbone(pretrained: bool = False) -> nn.Module:
    """
    Construct a ResNet-18 backbone adapted for single-channel medical imaging.

    Replaces initial 7x7 conv1 (in_channels=3) with in_channels=1,
    and replaces the final classification fc layer with Identity.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)

    if pretrained:
        # Average existing RGB filter weights across channel dimension
        orig_weight = backbone.conv1.weight.data
        new_conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        new_conv.weight.data = orig_weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv
    else:
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    # Replace 1000-class classifier with Identity to output 512-D spatial features
    backbone.fc = nn.Identity()
    return backbone


class DualEncoderResNet18(nn.Module):
    """
    Dual-Encoder ResNet-18 for 2D Ultrasound and 2D MRI cross-modal representation.

    Parameters
    ----------
    embedding_dim : int, optional
        Dimensionality of the projected latent embedding space D, by default 128.
    siamese : bool, optional
        If True, shares weights between ultrasound and MRI encoders to enforce
        strict modality-invariant representations during contrastive pre-training.
        By default True.
    pretrained : bool, optional
        Whether to initialize backbones with ImageNet weights, by default False.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        siamese: bool = True,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.siamese = siamese

        # Primary ultrasound encoder branch (E_US)
        self.encoder_us = _build_resnet18_backbone(pretrained=pretrained)
        self.projector_us = nn.Linear(512, embedding_dim)

        if siamese:
            # Shared weight configuration (Siamese mode)
            self.encoder_mr = self.encoder_us
            self.projector_mr = self.projector_us
        else:
            # Decoupled weight configuration
            self.encoder_mr = _build_resnet18_backbone(pretrained=pretrained)
            self.projector_mr = nn.Linear(512, embedding_dim)

    def decouple_weights(self) -> None:
        """
        Decouple shared Siamese weights into independent MRI and US branches.
        Used when transitioning from contrastive pre-training to downstream refinement.
        """
        if self.siamese:
            self.encoder_mr = copy.deepcopy(self.encoder_us)
            self.projector_mr = copy.deepcopy(self.projector_us)
            self.siamese = False

    def encode_us(
        self, us_tensor: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Extract ultrasound features or projected L2-normalized embeddings.

        Parameters
        ----------
        us_tensor : torch.Tensor
            Single-channel ultrasound tensor [Batch, 1, 227, 227].
        return_features : bool, optional
            If True, returns raw 512-D backbone features (for Stage 1 fusion).
            If False, returns normalized D=128 embedding.
        """
        features = self.encoder_us(us_tensor)
        if return_features:
            return features
        projected = self.projector_us(features)
        return F.normalize(projected, p=2, dim=-1)

    def encode_mr(
        self, mri_tensor: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Extract MRI features or projected L2-normalized embeddings.

        Parameters
        ----------
        mri_tensor : torch.Tensor
            Single-channel MRI slice tensor [Batch, 1, 227, 227].
        return_features : bool, optional
            If True, returns raw 512-D backbone features.
            If False, returns normalized D=128 embedding.
        """
        features = self.encoder_mr(mri_tensor)
        if return_features:
            return features
        projected = self.projector_mr(features)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        us_tensor: torch.Tensor,
        mri_tensor: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass processing paired ultrasound and MRI 2D slices.

        Parameters
        ----------
        us_tensor : torch.Tensor
            Batch of 2D ultrasound slices [Batch, 1, 227, 227].
        mri_tensor : torch.Tensor
            Batch of 2D MRI slices [Batch, 1, 227, 227].
        return_features : bool, optional
            If True, returns 512-D feature representations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            z_us : [Batch, embedding_dim] normalized embeddings.
            z_mr : [Batch, embedding_dim] normalized embeddings.
        """
        z_us = self.encode_us(us_tensor, return_features=return_features)
        z_mr = self.encode_mr(mri_tensor, return_features=return_features)
        return z_us, z_mr
