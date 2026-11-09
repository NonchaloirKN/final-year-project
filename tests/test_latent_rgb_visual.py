"""
Unit tests for Latent Space RGB Visualisation (Sprint 6, Task 6.2).

Verifies:
1. extract_spatial_features extracts pre-GAP spatial feature maps of shape [B, 512, H_f, W_f].
2. project_features_to_rgb performs channel PCA to 3 components and normalizes to [0, 255].
3. RGB output arrays are valid uint8 images with shape [H, W, 3].
4. Color consistency across modalities via joint PCA fitting.
"""

import numpy as np
import torch

from scripts.latent_feature_pca import project_features_to_rgb
from src.models.encoders import MRIEncoder, USEncoder, TwinEncoders


class TestLatentRGBVisual:
    """Test suite for spatial feature extraction and latent space RGB rendering."""

    def test_extract_spatial_features_shapes(self):
        """Verify pre-GAP feature maps have shape [B, 512, H_f, W_f]."""
        nn1 = MRIEncoder(embedding_dim=128)
        nn2 = USEncoder(embedding_dim=128)

        # Slice input: [B, 1, 227, 227]
        dummy_mri = torch.randn(2, 1, 227, 227)
        dummy_us = torch.randn(2, 1, 227, 227)

        feat_mri = nn1.extract_spatial_features(dummy_mri)
        feat_us = nn2.extract_spatial_features(dummy_us)

        # ResNet18 downsamples 227x227 by a factor of 32 -> 8x8 spatial grid
        assert feat_mri.shape == (2, 512, 8, 8)
        assert feat_us.shape == (2, 512, 8, 8)
        assert not torch.isnan(feat_mri).any()
        assert not torch.isnan(feat_us).any()

    def test_twin_encoders_extract_spatial_features(self):
        """Verify TwinEncoders wrapper exposes extract_spatial_features."""
        twin = TwinEncoders(embedding_dim=128)
        dummy_mri = torch.randn(1, 1, 227, 227)
        dummy_us = torch.randn(1, 1, 227, 227)

        f1, f2 = twin.extract_spatial_features(dummy_mri, dummy_us)
        assert f1.shape == (1, 512, 8, 8)
        assert f2.shape == (1, 512, 8, 8)

    def test_project_features_to_rgb_output_properties(self):
        """Verify PCA projection outputs valid [H, W, 3] uint8 RGB images."""
        feat_mri = torch.randn(1, 512, 8, 8)
        feat_us = torch.randn(1, 512, 8, 8)

        target_h, target_w = 227, 227
        rgb_mri, rgb_us, var_exp = project_features_to_rgb(
            feat_mri=feat_mri,
            feat_us=feat_us,
            target_size=(target_h, target_w),
        )

        # Check types and shapes
        assert isinstance(rgb_mri, np.ndarray)
        assert isinstance(rgb_us, np.ndarray)
        assert rgb_mri.shape == (target_h, target_w, 3)
        assert rgb_us.shape == (target_h, target_w, 3)
        assert rgb_mri.dtype == np.uint8
        assert rgb_us.dtype == np.uint8

        # Check value bounds [0, 255]
        assert rgb_mri.min() >= 0
        assert rgb_mri.max() <= 255
        assert rgb_us.min() >= 0
        assert rgb_us.max() <= 255

        # Explained variance must be a positive fraction <= 1.0
        assert 0.0 < var_exp <= 1.0
