"""
Continuous 6D Pose Regressor (NN3).

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Implements a multi-layer perceptron (MLP) regressor mapping 128-D latent embeddings
to a continuous 9D pose vector:
- 3 parameters for continuous translation [tx, ty, tz] (normalized in [-1, 1]).
- 6 parameters for continuous SO(3) rotation [a1, a2] (Zhou et al., CVPR 2019).

Architecture Modularity:
Features modular heads decoupling translation (with separate z_head and xy_head)
and continuous 6D rotation, reflecting Dylan Hellig's out-of-plane vs in-plane
decoupling insight (HLLDYL001).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous 6D rotation representation (Zhou et al., CVPR 2019)
    to valid 3x3 SO(3) rotation matrices using Gram-Schmidt orthogonalization.

    Args:
        d6: Tensor of shape (..., 6) where columns represent [a1, a2].

    Returns:
        R: Tensor of shape (..., 3, 3) representing orthogonal matrices with det(R) = +1.
    """
    orig_shape = d6.shape[:-1]
    d6_flat = d6.reshape(-1, 6)

    a1 = d6_flat[..., :3]
    a2 = d6_flat[..., 3:6]

    # First orthogonal column
    b1 = F.normalize(a1, p=2, dim=-1, eps=1e-8)

    # Gram-Schmidt orthogonalization for second column
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2_raw = a2 - dot * b1
    b2 = F.normalize(b2_raw, p=2, dim=-1, eps=1e-8)

    # Cross product for third column
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack column vectors into 3x3 matrix: [b1, b2, b3]
    R = torch.stack([b1, b2, b3], dim=-1)
    return R.reshape(*orig_shape, 3, 3)


class PoseRegressor(nn.Module):
    """
    NN3: Continuous 6D Pose Regressor.

    Maps a 128-D latent embedding to continuous 9D pose parameters
    [tx, ty, tz, a1_x, a1_y, a1_z, a2_x, a2_y, a2_z].
    """

    def __init__(
        self,
        in_features: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        symmetric_trunk: bool = True,
    ) -> None:
        """
        Args:
            in_features: Embedding dimension (128).
            hidden_dim: Trunk width (256).
            dropout: Dropout probability after each hidden block.
            symmetric_trunk: True (default) gives three identical Linear -> LayerNorm -> GELU blocks with
                dropout after the two hidden blocks (134,153 parameters). False rebuilds the Sprint 7 / 8a
                trunk (one dropout, no LayerNorm on the last block; 133,897 parameters) so that older
                checkpoints still load.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.symmetric_trunk = symmetric_trunk

        # Shared feature extraction trunk
        if symmetric_trunk:
            self.trunk = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
            )

        # Modular prediction heads (enables decoupling z-depth and tilt per Dylan Hellig's thesis)
        # Translation: xy (in-plane, 2 DoF) and z (out-of-plane depth/height, 1 DoF)
        self.xy_head = nn.Linear(128, 2)
        self.z_head = nn.Linear(128, 1)

        # Rotation: continuous 6D rotation representation (Gram-Schmidt SO(3))
        self.rot_head = nn.Linear(128, 6)

        # Initialize rotation head near identity: a1=[1,0,0], a2=[0,1,0]
        self._init_identity_rotation_head()

    def _init_identity_rotation_head(self) -> None:
        """Initialize the rotation head weights to output near identity rotation."""
        with torch.no_grad():
            # The symmetric trunk ends in LayerNorm, so h has O(1) entries; a 1e-3 weight scale keeps the
            # initial 6D output within ~1e-2 of identity (0.01 for the legacy trunk, as before)
            self.rot_head.weight.data.normal_(0.0, 0.001 if self.symmetric_trunk else 0.01)
            # Default bias [1, 0, 0, 0, 1, 0] -> identity matrix
            self.rot_head.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            if self.symmetric_trunk:
                # Translation heads start at the volume centre (the trivial predictor), for the same reason
                for head in (self.xy_head, self.z_head):
                    head.weight.data.normal_(0.0, 0.001)
                    head.bias.data.zero_()

    def predict_translation(self, h: torch.Tensor) -> torch.Tensor:
        """Predict 3D translation [tx, ty, tz] from trunk features."""
        t_xy = self.xy_head(h)
        t_z = self.z_head(h)
        return torch.cat([t_xy, t_z], dim=-1)

    def predict_rotation_6d(self, h: torch.Tensor) -> torch.Tensor:
        """Predict continuous 6D rotation [a1, a2] from trunk features."""
        return self.rot_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting 9D pose vector.

        Args:
            x: Latent embedding tensor of shape [B, in_features] (e.g. from NN1 or NN2).

        Returns:
            pose_9d: Tensor of shape [B, 9] containing [tx, ty, tz, a1_x..z, a2_x..z].
        """
        h = self.trunk(x)
        trans = self.predict_translation(h)
        rot_6d = self.predict_rotation_6d(h)
        return torch.cat([trans, rot_6d], dim=-1)

    def predict_pose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method predicting 3D translation and 3x3 rotation matrix.

        Args:
            x: Latent embedding tensor of shape [B, in_features].

        Returns:
            translation: [B, 3] continuous translation offsets.
            rotation_matrix: [B, 3, 3] orthogonal SO(3) rotation matrices.
        """
        pose_9d = self.forward(x)
        trans = pose_9d[..., :3]
        rot_mat = rotation_6d_to_matrix(pose_9d[..., 3:9])
        return trans, rot_mat

    @staticmethod
    def to_rotation_matrix(pose_9d: torch.Tensor) -> torch.Tensor:
        """Convert the rotation component of a 9D pose tensor to a 3x3 rotation matrix."""
        return rotation_6d_to_matrix(pose_9d[..., 3:9])


# Alias reflecting the technical brief naming
NN3 = PoseRegressor
