"""
Unit tests for 'Baby Steps' Incremental Training Modes (Sprint 6, Task 6.1).

Verifies:
1. MultiModalPoseLoss with train_translation_only=True ignores rotation error and isolates translation loss.
2. MultiModalPoseLoss with train_rotation_only=True ignores translation error and isolates rotation loss.
3. Default Joint Loss computes composite pose loss combining both translation and rotation.
4. Gradients flow exclusively to active components during backward pass.
"""

import math
import torch
import torch.nn as nn

from src.losses.pose_loss import MultiModalPoseLoss


class TestIncrementalTrainingFlags:
    """Test suite verifying decoupled incremental pose loss logic."""

    def test_translation_only_ignores_rotation_error(self):
        """When train_translation_only=True, rotation discrepancies must not increase pose loss."""
        criterion = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, train_translation_only=True)

        # Ground truth pose: translation [0, 0, 0], identity rotation
        p_gt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        # Prediction A: translation matches perfectly, but rotation is 90 degrees off (Frobenius = 4.0)
        p_pred_rot_error = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])

        # Prediction B: perfect translation AND perfect rotation
        p_pred_perfect = p_gt.clone()

        loss_a, details_a = criterion(p_pred=p_pred_rot_error, p_gt=p_gt, return_details=True)
        loss_b, details_b = criterion(p_pred=p_pred_perfect, p_gt=p_gt, return_details=True)

        # Total loss must be 0.0 in both cases because rotation is ignored
        assert torch.isclose(loss_a, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(loss_b, torch.tensor(0.0), atol=1e-6)
        assert math.isclose(details_a["pose_loss"], 0.0, abs_tol=1e-6)
        assert math.isclose(details_a["translation_loss"], 0.0, abs_tol=1e-6)
        # Rotation loss is recorded in details for monitoring, but not included in total
        assert details_a["rotation_loss"] > 1.0

    def test_translation_only_penalizes_translation_error(self):
        """When train_translation_only=True, translation errors scale loss proportionally."""
        criterion = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, train_translation_only=True)

        p_gt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        # Translation offset [0.3, 0.4, 0.0] -> MSE = (0.09 + 0.16) / 3 = 0.25 / 3 = 0.08333
        p_pred = torch.tensor([[0.3, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        loss, details = criterion(p_pred=p_pred, p_gt=p_gt, return_details=True)
        expected_mse = (0.3**2 + 0.4**2) / 3.0
        assert math.isclose(loss.item(), expected_mse, abs_tol=1e-5)
        assert math.isclose(details["translation_loss"], expected_mse, abs_tol=1e-5)

    def test_rotation_only_ignores_translation_error(self):
        """When train_rotation_only=True, translation discrepancies must not increase pose loss."""
        criterion = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, train_rotation_only=True)

        p_gt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        # Prediction has massive translation error, but identity rotation
        p_pred_trans_error = torch.tensor([[10.0, -5.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        loss, details = criterion(p_pred=p_pred_trans_error, p_gt=p_gt, return_details=True)

        # Pose loss must be 0.0 because rotation error is zero
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
        assert math.isclose(details["pose_loss"], 0.0, abs_tol=1e-6)
        assert math.isclose(details["rotation_loss"], 0.0, abs_tol=1e-6)
        # Translation loss is monitored but excluded
        assert details["translation_loss"] > 10.0

    def test_rotation_only_penalizes_rotation_error(self):
        """When train_rotation_only=True, rotation errors scale loss via Chordal Frobenius."""
        criterion = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, train_rotation_only=True)

        I = torch.eye(3).unsqueeze(0)
        # 90-degree Z rotation -> Frobenius squared error = 4.0
        R_90 = torch.tensor([[[0.0, -1.0, 0.0],
                              [1.0,  0.0, 0.0],
                              [0.0,  0.0, 1.0]]])

        t_dummy = torch.zeros(1, 3)
        loss, details = criterion(p_pred=(t_dummy, R_90), p_gt=(t_dummy, I), return_details=True)

        assert math.isclose(loss.item(), 4.0, abs_tol=1e-5)
        assert math.isclose(details["rotation_loss"], 4.0, abs_tol=1e-5)
        assert math.isclose(details["pose_loss"], 4.0, abs_tol=1e-5)

    def test_joint_loss_default_combines_both(self):
        """When neither flag is set, both translation and rotation contribute to pose loss."""
        criterion = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0)
        assert not criterion.train_translation_only
        assert not criterion.train_rotation_only

        p_gt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        # Translation offset [0.3, 0.0, 0.0] -> MSE = 0.09 / 3 = 0.03
        # Rotation 90 deg -> Frobenius = 4.0
        p_pred = torch.tensor([[0.3, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])

        loss, details = criterion(p_pred=p_pred, p_gt=p_gt, return_details=True)
        expected_total = 0.03 + 4.0  # 4.03
        assert math.isclose(loss.item(), expected_total, abs_tol=1e-4)
        assert math.isclose(details["translation_loss"], 0.03, abs_tol=1e-4)
        assert math.isclose(details["rotation_loss"], 4.0, abs_tol=1e-4)

    def test_gradient_isolation_during_backprop(self):
        """
        Verify that in train_translation_only mode, rotation prediction parameters
        receive zero gradient, and vice versa in train_rotation_only mode.
        """
        # Linear layer outputting 9D pose from dummy latent vector
        linear_head = nn.Linear(16, 9)
        x = torch.randn(2, 16)

        p_gt = torch.zeros(2, 9)
        p_gt[:, 3] = 1.0
        p_gt[:, 7] = 1.0

        # Case 1: Translation Only
        criterion_t = MultiModalPoseLoss(train_translation_only=True)
        p_pred = linear_head(x)
        loss_t = criterion_t(p_pred=p_pred, p_gt=p_gt)

        linear_head.zero_grad()
        loss_t.backward()

        # Weight rows corresponding to translation (indices 0, 1, 2) must have non-zero gradients
        # Weight rows corresponding to rotation (indices 3..8) must have strictly ZERO gradients
        assert torch.any(linear_head.weight.grad[:3] != 0.0), "Translation weights must receive gradients"
        assert torch.all(linear_head.weight.grad[3:] == 0.0), "Rotation weights must receive ZERO gradients in trans-only mode"

        # Case 2: Rotation Only
        linear_head.zero_grad()
        criterion_r = MultiModalPoseLoss(train_rotation_only=True)
        p_pred_2 = linear_head(x)
        loss_r = criterion_r(p_pred=p_pred_2, p_gt=p_gt)
        loss_r.backward()

        assert torch.all(linear_head.weight.grad[:3] == 0.0), "Translation weights must receive ZERO gradients in rot-only mode"
        assert torch.any(linear_head.weight.grad[3:] != 0.0), "Rotation weights must receive gradients in rot-only mode"
