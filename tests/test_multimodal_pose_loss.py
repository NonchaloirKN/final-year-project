"""
Unit tests for MultiModalPoseLoss.

Asserts:
1. Identical embeddings and perfect pose prediction produce exactly 0.0 total loss.
2. Chordal Frobenius rotation loss matches mathematical formula 8 * sin^2(theta / 2).
3. Translation and rotation errors scale loss monotonically.
4. Gradients flow backwards without NaNs into embeddings and predicted poses.
5. Dual-prediction (MRI branch + US branch) and single-prediction modes operate correctly.
6. Component loss breakdown dictionary is correctly returned.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from src.losses.pose_loss import MultiModalPoseLoss, chordal_frobenius_loss, symmetric_info_nce
from src.models.pose_regressor import rotation_6d_to_matrix


def test_multimodal_pose_loss_zero_on_identical_inputs():
    """Verify total loss is exactly 0.0 when embeddings and poses match perfectly."""
    criterion = MultiModalPoseLoss(alpha=1.0)

    # Identical embeddings
    emb1 = F.normalize(torch.randn(4, 128), p=2, dim=-1)
    emb2 = emb1.clone()

    # Ground truth 9D pose
    p_gt = torch.zeros(4, 9)
    p_gt[:, :3] = torch.tensor([0.2, -0.1, 0.5])
    p_gt[:, 3] = 1.0  # a1 = [1, 0, 0]
    p_gt[:, 7] = 1.0  # a2 = [0, 1, 0]
    p_pred = p_gt.clone()

    loss, details = criterion(emb1, emb2, p_pred, p_gt, return_details=True)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected 0.0 loss, got {loss.item()}"
    assert details["modality_loss"] == pytest.approx(0.0, abs=1e-6)
    assert details["translation_loss"] == pytest.approx(0.0, abs=1e-6)
    assert details["rotation_loss"] == pytest.approx(0.0, abs=1e-6)


def test_chordal_frobenius_loss_properties():
    """Verify Chordal Frobenius loss matches 8 * sin^2(theta / 2)."""
    # 1. Identity rotation vs Identity rotation -> 0.0
    I = torch.eye(3).unsqueeze(0)
    assert torch.isclose(chordal_frobenius_loss(I, I), torch.tensor(0.0), atol=1e-6)

    # 2. 90-degree rotation around Z axis (theta = pi/2)
    # Expected Frobenius loss: 8 * sin^2(pi / 4) = 8 * (1 / sqrt(2))^2 = 8 * 0.5 = 4.0
    R_rot90 = torch.tensor([[[0.0, -1.0, 0.0],
                             [1.0,  0.0, 0.0],
                             [0.0,  0.0, 1.0]]])
    loss_90 = chordal_frobenius_loss(R_rot90, I)
    assert torch.isclose(loss_90, torch.tensor(4.0), atol=1e-5), f"Expected 4.0, got {loss_90.item()}"

    # 3. 180-degree rotation around Z axis (theta = pi)
    # Expected Frobenius loss: 8 * sin^2(pi / 2) = 8 * 1.0 = 8.0
    R_rot180 = torch.tensor([[[-1.0,  0.0, 0.0],
                              [ 0.0, -1.0, 0.0],
                              [ 0.0,  0.0, 1.0]]])
    loss_180 = chordal_frobenius_loss(R_rot180, I)
    assert torch.isclose(loss_180, torch.tensor(8.0), atol=1e-5), f"Expected 8.0, got {loss_180.item()}"


def test_multimodal_pose_loss_gradient_flow():
    """Verify autograd backpropagation yields non-NaN gradients across all inputs."""
    criterion = MultiModalPoseLoss(alpha=1.0)

    raw_emb1 = torch.randn(3, 128, requires_grad=True)
    raw_emb2 = torch.randn(3, 128, requires_grad=True)
    emb1 = F.normalize(raw_emb1, p=2, dim=-1)
    emb2 = F.normalize(raw_emb2, p=2, dim=-1)
    p_pred = torch.randn(3, 9, requires_grad=True)

    p_gt = torch.zeros(3, 9)
    p_gt[:, 3] = 1.0
    p_gt[:, 7] = 1.0

    loss = criterion(emb1, emb2, p_pred, p_gt)
    loss.backward()

    assert raw_emb1.grad is not None
    assert raw_emb2.grad is not None
    assert p_pred.grad is not None
    assert not torch.isnan(raw_emb1.grad).any()
    assert not torch.isnan(raw_emb2.grad).any()
    assert not torch.isnan(p_pred.grad).any()
    assert (p_pred.grad != 0).any()


def test_multimodal_pose_loss_dual_mri_us_predictions():
    """Verify loss formulation when both MRI and US pose predictions are supplied."""
    criterion = MultiModalPoseLoss(alpha=1.0)

    emb1 = F.normalize(torch.randn(2, 128), p=2, dim=-1)
    emb2 = F.normalize(torch.randn(2, 128), p=2, dim=-1)
    p_pred_us = torch.randn(2, 9)
    p_pred_mri = torch.randn(2, 9)

    p_gt = torch.zeros(2, 9)
    p_gt[:, 3] = 1.0
    p_gt[:, 7] = 1.0

    loss, details = criterion(emb1, emb2, p_pred_us, p_gt, p_pred_mri=p_pred_mri, return_details=True)

    assert loss.item() > 0.0
    assert "total_loss" in details
    assert "modality_loss" in details
    assert "translation_loss" in details
    assert "rotation_loss" in details
    assert "pose_loss" in details


def test_translation_scale_balances_normalised_residuals():
    """A 20 mm error on a 256 mm axis (0.157 normalised) must cost 1.0 with trans_scale = half_extent / 20."""
    p_gt = torch.zeros(1, 9)
    p_gt[:, 3] = 1.0
    p_gt[:, 7] = 1.0
    p_pred = p_gt.clone()
    p_pred[:, 0] = 20.0 / 127.5

    legacy = MultiModalPoseLoss(alpha=1.0)
    _, d_legacy = legacy(p_pred=p_pred, p_gt=p_gt, return_details=True)
    assert d_legacy["translation_loss"] == pytest.approx((20.0 / 127.5) ** 2 / 3.0, rel=1e-5)

    scale = (127.5 / 20.0, 127.5 / 20.0, 95.5 / 20.0)
    scaled = MultiModalPoseLoss(alpha=1.0, trans_scale=scale)
    _, d_scaled = scaled(p_pred=p_pred, p_gt=p_gt, return_details=True)
    assert d_scaled["translation_loss"] == pytest.approx(1.0 / 3.0, rel=1e-5)

    # Per-call override matches the constructor setting
    _, d_override = legacy(p_pred=p_pred, p_gt=p_gt, trans_scale=scale, return_details=True)
    assert d_override["translation_loss"] == pytest.approx(d_scaled["translation_loss"], rel=1e-6)


def test_contrastive_term_repels_collapsed_embeddings():
    """InfoNCE is ln(B) for a collapsed cone and near zero for well-separated matched pairs."""
    torch.manual_seed(0)
    b = 8
    collapsed = F.normalize(torch.ones(b, 128) + 1e-3 * torch.randn(b, 128), dim=-1)
    assert symmetric_info_nce(collapsed, collapsed.clone(), 0.1).item() == pytest.approx(math.log(b), abs=0.05)

    separated = torch.eye(b, 128)
    assert symmetric_info_nce(separated, separated.clone(), 0.1).item() < 1e-3

    p_gt = torch.zeros(b, 9)
    p_gt[:, 3] = 1.0
    p_gt[:, 7] = 1.0
    crit = MultiModalPoseLoss(alpha=1.0, contrastive_weight=0.1)
    _, d_col = crit(collapsed, collapsed.clone(), p_gt.clone(), p_gt, return_details=True)
    _, d_sep = crit(separated, separated.clone(), p_gt.clone(), p_gt, return_details=True)
    assert d_col["contrastive_loss"] > d_sep["contrastive_loss"] + 1.0
    assert d_col["modality_loss"] == pytest.approx(0.1 * d_col["contrastive_loss"], abs=1e-4)


def test_axis_weights_emphasise_elevation():
    """axis_weights scale the squared residual per axis: a z error costs 1.5x an equal x error."""
    p_gt = torch.zeros(1, 9)
    p_gt[:, 3] = 1.0
    p_gt[:, 7] = 1.0
    px, pz = p_gt.clone(), p_gt.clone()
    px[:, 0] = 0.1
    pz[:, 2] = 0.1
    crit = MultiModalPoseLoss(alpha=1.0, axis_weights=(1.0, 1.0, 1.5))
    _, dx = crit(p_pred=px, p_gt=p_gt, return_details=True)
    _, dz = crit(p_pred=pz, p_gt=p_gt, return_details=True)
    assert dz["translation_loss"] == pytest.approx(1.5 * dx["translation_loss"], rel=1e-5)
