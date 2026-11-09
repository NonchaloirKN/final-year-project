"""
Twin Feature Encoders (NN1 and NN2) for Continuous 6D Pose Estimation.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Implements deep convolutional feature extractors:
- NN1 (MRIEncoder): 2D ResNet-18 modified for single-channel MRI slices.
- NN2 (USEncoder): 2D ResNet-18 modified for single-channel Ultrasound slices.
Both encoders map 2D slices to an L2-normalized 128-dimensional latent space S^(127)
acting as a modality-invariant 'primal sketch'.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def _build_single_channel_resnet18(pretrained: bool = False) -> nn.Module:
    """
    Construct a ResNet-18 backbone adapted for single-channel medical imaging.
    Replaces conv1 (in_channels=3) with in_channels=1 and replaces fc with nn.Identity().
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)

    if pretrained:
        orig_weight = backbone.conv1.weight.data
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = orig_weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv
    else:
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    backbone.fc = nn.Identity()
    return backbone


class MRIEncoder(nn.Module):
    """
    NN1: Feature encoder for 2D MRI slices.

    Maps a 2D scalar MRI slice [B, 1, H, W] to an L2-normalized embedding in R^D (D=128).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = _build_single_channel_resnet18(pretrained=pretrained)
        self.projector = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass for MRI slices.

        Args:
            x: Input MRI slice tensor of shape [B, 1, H, W].
            return_features: If True, returns unnormalized 512-D spatial features.

        Returns:
            Embedding tensor of shape [B, embedding_dim] with unit L2 norm.
        """
        features = self.backbone(x)
        if return_features:
            return features
        projected = self.projector(features)
        return F.normalize(projected, p=2, dim=-1)

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 2D spatial feature map before Global Average Pooling.

        Args:
            x: Input slice tensor of shape [B, 1, H, W].

        Returns:
            Spatial feature tensor of shape [B, 512, H_feat, W_feat]
            (e.g., [B, 512, 8, 8] for 227x227 inputs).
        """
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return x


class USEncoder(nn.Module):
    """
    NN2: Feature encoder for 2D Ultrasound slices.

    Maps a 2D scalar Ultrasound slice [B, 1, H, W] to an L2-normalized embedding in R^D (D=128).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = _build_single_channel_resnet18(pretrained=pretrained)
        self.projector = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass for Ultrasound slices.

        Args:
            x: Input Ultrasound slice tensor of shape [B, 1, H, W].
            return_features: If True, returns unnormalized 512-D spatial features.

        Returns:
            Embedding tensor of shape [B, embedding_dim] with unit L2 norm.
        """
        features = self.backbone(x)
        if return_features:
            return features
        projected = self.projector(features)
        return F.normalize(projected, p=2, dim=-1)

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 2D spatial feature map before Global Average Pooling.

        Args:
            x: Input slice tensor of shape [B, 1, H, W].

        Returns:
            Spatial feature tensor of shape [B, 512, H_feat, W_feat]
            (e.g., [B, 512, 8, 8] for 227x227 inputs).
        """
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return x


# Aliases reflecting the technical brief naming
NN1 = MRIEncoder
NN2 = USEncoder


class TwinEncoders(nn.Module):
    """
    Twin Encoders wrapper encapsulating NN1 (MRI) and NN2 (US).

    Supports independent branches, shared Siamese weights, and loading
    from contrastive pre-trained checkpoints.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        shared_weights: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.shared_weights = shared_weights

        self.nn1 = MRIEncoder(embedding_dim=embedding_dim, pretrained=pretrained)
        if shared_weights:
            self.nn2 = self.nn1  # Share identical weights
        else:
            self.nn2 = USEncoder(embedding_dim=embedding_dim, pretrained=pretrained)

    def decouple_weights(self) -> None:
        """Decouple shared Siamese weights into independent MRI and US branches."""
        if self.shared_weights:
            self.nn2 = copy.deepcopy(self.nn1)
            self.shared_weights = False

    def load_contrastive_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load weights from a Sprint 2 DualEncoderResNet18 contrastive checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Map state dict keys from DualEncoderResNet18 to NN1/NN2
        nn1_dict = {}
        nn2_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder_mr."):
                nn1_dict[k.replace("encoder_mr.", "backbone.")] = v
            elif k.startswith("projector_mr."):
                nn1_dict[k.replace("projector_mr.", "projector.")] = v
            elif k.startswith("encoder_us."):
                nn2_dict[k.replace("encoder_us.", "backbone.")] = v
            elif k.startswith("projector_us."):
                nn2_dict[k.replace("projector_us.", "projector.")] = v

        if nn1_dict:
            self.nn1.load_state_dict(nn1_dict, strict=False)
        if nn2_dict and not self.shared_weights:
            self.nn2.load_state_dict(nn2_dict, strict=False)

    def forward(
        self,
        mri_slice: torch.Tensor,
        us_slice: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract normalized embeddings for paired MRI and US slices.

        Args:
            mri_slice: [B, 1, H, W]
            us_slice:  [B, 1, H, W]

        Returns:
            emb1: [B, embedding_dim] L2-normalized MRI embedding.
            emb2: [B, embedding_dim] L2-normalized US embedding.
        """
        emb1 = self.nn1(mri_slice, return_features=return_features)
        emb2 = self.nn2(us_slice, return_features=return_features)
        return emb1, emb2

    def extract_spatial_features(
        self,
        mri_slice: torch.Tensor,
        us_slice: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pre-GAP spatial feature maps for paired MRI and US slices.

        Args:
            mri_slice: [B, 1, H, W]
            us_slice:  [B, 1, H, W]

        Returns:
            feat1: [B, 512, H_feat, W_feat] spatial MRI feature map.
            feat2: [B, 512, H_feat, W_feat] spatial US feature map.
        """
        feat1 = self.nn1.extract_spatial_features(mri_slice)
        feat2 = self.nn2.extract_spatial_features(us_slice)
        return feat1, feat2
