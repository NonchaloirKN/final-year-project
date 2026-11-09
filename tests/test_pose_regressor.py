"""
Unit tests for Continuous 6D Pose Regressor (NN3).

Asserts:
1. Output tensor shape is strictly [B, 9] (3 translation + 6 rotation).
2. The 6D rotation component converts to a strictly orthogonal 3x3 SO(3) matrix:
   R * R^T = I and det(R) = +1.0.
3. predict_pose(x) returns valid [B, 3] translations and [B, 3, 3] rotation matrices.
4. Gradients flow backwards through trunk and modular heads without NaNs.
5. Modular heads (z_head, xy_head, rot_head) can be evaluated independently.
"""

import pytest
import torch

from src.models.pose_regressor import NN3, PoseRegressor, rotation_6d_to_matrix


def test_pose_regressor_output_shape():
    """Verify PoseRegressor outputs shape [B, 9]."""
    regressor = PoseRegressor(in_features=128)
    regressor.eval()

    batch_size = 5
    embeddings = torch.randn(batch_size, 128)
    with torch.no_grad():
        pose_9d = regressor(embeddings)

    assert pose_9d.shape == (batch_size, 9), f"Expected shape ({batch_size}, 9), got {pose_9d.shape}"


def test_pose_regressor_rotation_matrix_orthogonality():
    """Verify that 6D rotation component produces strictly orthogonal SO(3) matrices."""
    regressor = PoseRegressor(in_features=128)
    regressor.eval()

    batch_size = 8
    embeddings = torch.randn(batch_size, 128)
    with torch.no_grad():
        _, R = regressor.predict_pose(embeddings)

    assert R.shape == (batch_size, 3, 3)

    # Assert R * R^T = I
    I = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
    RtR = torch.bmm(R, R.transpose(1, 2))
    assert torch.allclose(RtR, I, atol=1e-5), f"Matrix not orthogonal! Diff: {(RtR - I).abs().max()}"

    # Assert det(R) = +1.0 (proper rotation, no reflection)
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(batch_size), atol=1e-5), f"Determinants not +1.0: {dets}"


def test_pose_regressor_identity_initialization():
    """Verify rotation head is initialized near identity."""
    regressor = PoseRegressor(in_features=128)
    # Zero input embedding
    zero_emb = torch.zeros(1, 128)
    _, R = regressor.predict_pose(zero_emb)

    # Initial rotation should be very close to identity
    I = torch.eye(3).unsqueeze(0)
    assert torch.allclose(R, I, atol=0.05), "Initial rotation deviates significantly from identity!"


def test_pose_regressor_gradient_flow():
    """Verify gradients propagate backwards through the regressor to the input."""
    regressor = PoseRegressor(in_features=128)
    regressor.train()

    embeddings = torch.randn(4, 128, requires_grad=True)
    pose_9d = regressor(embeddings)
    loss = pose_9d.sum()
    loss.backward()

    assert embeddings.grad is not None
    assert not torch.isnan(embeddings.grad).any()
    assert (embeddings.grad != 0).any()

    for name, param in regressor.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"


def test_modular_heads_decoupling():
    """
    Verify Dylan Hellig's decoupling: z_head, xy_head, and rot_head can be isolated.
    """
    regressor = PoseRegressor(in_features=128)
    regressor.eval()

    embeddings = torch.randn(3, 128)
    with torch.no_grad():
        h = regressor.trunk(embeddings)
        z_pred = regressor.z_head(h)
        xy_pred = regressor.xy_head(h)
        rot_pred = regressor.rot_head(h)

    assert z_pred.shape == (3, 1), f"Expected z_head shape (3, 1), got {z_pred.shape}"
    assert xy_pred.shape == (3, 2), f"Expected xy_head shape (3, 2), got {xy_pred.shape}"
    assert rot_pred.shape == (3, 6), f"Expected rot_head shape (3, 6), got {rot_pred.shape}"


def test_alias_nn3():
    """Verify NN3 alias points to PoseRegressor."""
    assert NN3 is PoseRegressor


def test_symmetric_and_legacy_trunk_parameter_counts():
    """Symmetric trunk (default) has 134,153 parameters; the legacy trunk keeps 133,897 for old checkpoints."""
    assert sum(p.numel() for p in PoseRegressor().parameters()) == 134153
    assert sum(p.numel() for p in PoseRegressor(symmetric_trunk=False).parameters()) == 133897
