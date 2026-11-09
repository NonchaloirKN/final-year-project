"""
Unit tests for physical 6D pose evaluation metrics (Sprint 5, Task 5.2).

Verifies mathematical precision and robustness of:
- compute_mte (Mean Translation Error in mm)
- compute_mae (Mean Angular Error in degrees)
- evaluate_pose_predictions (composite clinical evaluation)
"""

import math
import numpy as np
import pytest
import torch

from src.models.pose_regressor import rotation_6d_to_matrix
from src.utils.metrics import compute_mae, compute_mte, evaluate_pose_predictions, volume_half_extent_mm
from src.utils.reslicer import axis_angle_to_matrix, matrix_to_rotation_6d, physical_to_normalized_pose


class TestPoseEvaluationMetrics:
    """Test suite verifying 6D pose evaluation metrics."""

    def test_mte_zero_error_identical_translations(self):
        """Identical translation tensors should yield exactly 0.0 mm translation error."""
        t_gt = torch.zeros(4, 3)
        t_pred = torch.zeros(4, 3)
        mte = compute_mte(t_pred, t_gt, trans_bound_mm=20.0)
        assert isinstance(mte, torch.Tensor)
        assert torch.isclose(mte, torch.tensor(0.0), atol=1e-6)

    def test_mte_known_translation_offsets(self):
        """
        Verify known physical translation offsets in mm:
        - [0.5, 0, 0] with 20mm bounds -> 10.0 mm error.
        - [0.15, 0.20, 0] with 20mm bounds -> physical [3.0, 4.0, 0] -> 5.0 mm error.
        """
        # Case 1: 1D offset
        t_pred_1d = torch.tensor([[0.5, 0.0, 0.0]])
        t_gt_1d = torch.tensor([[0.0, 0.0, 0.0]])
        mte_1d = compute_mte(t_pred_1d, t_gt_1d, trans_bound_mm=20.0)
        assert torch.isclose(mte_1d, torch.tensor(10.0), atol=1e-5)

        # Case 2: 3D Pythagorean offset (3-4-0 -> 5)
        t_pred_3d = torch.tensor([[0.15, 0.20, 0.0]])
        t_gt_3d = torch.tensor([[0.0, 0.0, 0.0]])
        mte_3d = compute_mte(t_pred_3d, t_gt_3d, trans_bound_mm=20.0)
        assert torch.isclose(mte_3d, torch.tensor(5.0), atol=1e-5)

        # Case 3: Already in physical mm (is_normalized=False)
        t_pred_phys = torch.tensor([[3.0, 4.0, 0.0]])
        t_gt_phys = torch.tensor([[0.0, 0.0, 0.0]])
        mte_phys = compute_mte(t_pred_phys, t_gt_phys, is_normalized=False)
        assert torch.isclose(mte_phys, torch.tensor(5.0), atol=1e-5)

    def test_mte_full_pose_tensor_and_numpy(self):
        """Verify compute_mte slices full 9D pose tensors and handles numpy arrays."""
        p_pred = np.array([
            [0.15, 0.20, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        p_gt = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        # Sample 1: sqrt((0.15*20)^2 + (0.20*20)^2) = 5.0 mm
        # Sample 2: 0.25*20 = 5.0 mm
        # Mean = 5.0 mm
        mte_val = compute_mte(p_pred, p_gt, trans_bound_mm=20.0, reduction="mean")
        assert isinstance(mte_val, float)
        assert pytest.approx(mte_val, abs=1e-5) == 5.0

        # Per-sample errors
        err_arr = compute_mte(p_pred, p_gt, trans_bound_mm=20.0, reduction="none")
        assert isinstance(err_arr, np.ndarray)
        assert err_arr.shape == (2,)
        np.testing.assert_allclose(err_arr, [5.0, 5.0], atol=1e-5)

    def test_mae_zero_error_identity_rotation(self):
        """Identical rotation matrices should yield exactly 0.0 degrees angular error."""
        eye = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        mae = compute_mae(eye, eye)
        assert isinstance(mae, torch.Tensor)
        assert torch.isclose(mae, torch.tensor(0.0), atol=1e-5)

    def test_mae_known_rotations(self):
        """
        Verify known canonical 3D rotation angles:
        - 90 degrees around Z axis -> exactly 90.0 deg.
        - 45 degrees around X axis -> exactly 45.0 deg.
        - 180 degrees around Y axis -> exactly 180.0 deg.
        """
        eye = torch.eye(3).unsqueeze(0)

        # 90 deg around Z
        R_z_90 = torch.tensor([[
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ]])
        mae_90 = compute_mae(R_z_90, eye)
        assert torch.isclose(mae_90, torch.tensor(90.0), atol=1e-4)

        # 45 deg around X
        c45 = math.cos(math.radians(45.0))
        s45 = math.sin(math.radians(45.0))
        R_x_45 = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.0, c45, -s45],
            [0.0, s45,  c45],
        ]])
        mae_45 = compute_mae(R_x_45, eye)
        assert torch.isclose(mae_45, torch.tensor(45.0), atol=1e-4)

        # 180 deg around Y
        R_y_180 = torch.tensor([[
            [-1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 0.0, 0.0, -1.0],
        ]])
        mae_180 = compute_mae(R_y_180, eye)
        assert torch.isclose(mae_180, torch.tensor(180.0), atol=1e-4)

    def test_mae_6d_and_9d_continuous_representations(self):
        """Verify compute_mae automatically parses 6D and 9D representations."""
        # Create known 30 deg rotation matrix around axis [1, 1, 1] / sqrt(3)
        axis = torch.tensor([[1.0, 1.0, 1.0]]) / math.sqrt(3.0)
        angle = math.radians(30.0)
        axis_angle = axis * angle
        R_gt = axis_angle_to_matrix(axis_angle)

        # Convert to continuous 6D and composite 9D
        d6 = matrix_to_rotation_6d(R_gt)
        p_9d_pred = torch.cat([torch.tensor([[0.1, 0.1, 0.1]]), d6], dim=-1)
        p_9d_gt = torch.cat([torch.tensor([[0.0, 0.0, 0.0]]), matrix_to_rotation_6d(torch.eye(3).unsqueeze(0))], dim=-1)

        mae = compute_mae(p_9d_pred, p_9d_gt)
        assert torch.isclose(mae, torch.tensor(30.0), atol=1e-4)

    def test_mae_clamp_numerical_stability(self):
        """Clamp should prevent NaN when trace produces floating point values slightly outside [-1, 3]."""
        # Perturbed identity with trace = 3.000001
        R_perturbed = torch.eye(3).unsqueeze(0) * 1.0000005
        mae = compute_mae(R_perturbed, torch.eye(3).unsqueeze(0))
        assert not torch.isnan(mae)
        assert not torch.isinf(mae)
        assert torch.isclose(mae, torch.tensor(0.0), atol=1e-3)

    def test_evaluate_pose_predictions_helper(self):
        """Verify the composite evaluation helper outputs all required clinical metrics."""
        p_pred = torch.tensor([[0.15, 0.20, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        p_gt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        metrics = evaluate_pose_predictions(p_pred, p_gt, trans_bound_mm=20.0)
        assert "mte_mm" in metrics
        assert "mae_deg" in metrics
        assert "tx_err_mm" in metrics
        assert "ty_err_mm" in metrics
        assert "tz_err_mm" in metrics
        assert pytest.approx(metrics["mte_mm"], abs=1e-4) == 5.0
        assert pytest.approx(metrics["mae_deg"], abs=1e-4) == 0.0
        assert pytest.approx(metrics["tx_err_mm"], abs=1e-4) == 3.0
        assert pytest.approx(metrics["ty_err_mm"], abs=1e-4) == 4.0
        assert pytest.approx(metrics["tz_err_mm"], abs=1e-4) == 0.0

    def test_mte_true_millimetres_for_volume_normalised_poses(self):
        """
        Poses from sample_random_6d_poses are normalised by the volume half-extent, so MTE must be
        un-normalised with volume_half_extent_mm, not the 20 mm sampling bound. A zero (volume
        centre) prediction against +/-20 mm poses must give the true ~19.2 mm mean error.
        """
        shape, spacing = (192, 256, 256), (1.0, 1.0, 1.0)
        half = volume_half_extent_mm(shape, spacing)
        assert half == pytest.approx((127.5, 127.5, 95.5))

        gen = torch.Generator().manual_seed(0)
        t_mm = (torch.rand(4096, 3, generator=gen) * 2.0 - 1.0) * 20.0
        pose_mm = torch.cat([t_mm, torch.zeros(4096, 3)], dim=-1)
        t_norm = physical_to_normalized_pose(pose_mm, shape, spacing)[..., :3]
        zero = torch.zeros_like(t_norm)

        # Round trip: each per-slice error equals the physical offset magnitude
        err = compute_mte(zero, t_norm, trans_bound_mm=half, reduction="none")
        assert torch.allclose(err, t_mm.norm(dim=-1), atol=1e-3)

        # Mean distance of a uniform point in a +/-20 mm cube from its centre is 0.9606 * 20 mm
        mte = compute_mte(zero, t_norm, trans_bound_mm=half)
        assert 18.7 < mte.item() < 19.7

        # The legacy 20 mm scaling understates the same error by ~5-6x
        legacy = compute_mte(zero, t_norm, trans_bound_mm=20.0)
        assert legacy.item() < 4.0

        # evaluate_pose_predictions accepts the per-axis tuple too
        p_gt = torch.zeros(4096, 9)
        p_gt[:, :3] = t_norm
        p_gt[:, 3] = 1.0
        p_gt[:, 7] = 1.0
        p_zero = p_gt.clone()
        p_zero[:, :3] = 0.0
        metrics = evaluate_pose_predictions(p_zero, p_gt, trans_bound_mm=half)
        assert pytest.approx(metrics["mte_mm"], abs=1e-3) == mte.item()
        assert pytest.approx(metrics["tz_err_mm"], abs=0.3) == 10.0
