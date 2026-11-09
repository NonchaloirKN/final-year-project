"""
Unit tests for Pre-Operative Patient Calibration Protocol (Sprint 5, Task 5.1).

Verifies:
1. freeze_encoders sets requires_grad = False for all parameters of NN1 and NN2.
2. unfreeze_regressor sets requires_grad = True for all parameters of NN3.
3. Gradients on NN1 and NN2 remain None or strictly zero after backward pass.
4. Gradients on NN3 are non-zero and populated.
5. NN1 and NN2 weights remain 100% bit-exact identical before and after optimizer steps,
   while NN3 weights update and specialize.
6. Calibration evaluation function accurately reports MTE and MAE.
7. End-to-end dry-run calibration executes cleanly.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from scripts.calibrate_patient import (
    calibrate_patient,
    evaluate_patient_calibration,
    freeze_encoders,
    unfreeze_regressor,
)
from src.losses.pose_loss import MultiModalPoseLoss
from src.models.encoders import NN1, NN2
from src.models.pose_regressor import NN3
from src.utils.reslicer import VolumeReslicer


class TestPatientCalibrationProtocol:
    """Test suite verifying freezing logic and patient calibration specialization."""

    def test_encoder_freezing_flags(self):
        """Verify requires_grad is False for NN1/NN2 and True for NN3."""
        nn1 = NN1(embedding_dim=128)
        nn2 = NN2(embedding_dim=128)
        nn3 = NN3(in_features=128)

        frozen_1, frozen_2 = freeze_encoders(nn1, nn2)
        trainable_3 = unfreeze_regressor(nn3)

        assert frozen_1 > 0
        assert frozen_2 > 0
        assert trainable_3 > 0

        # All NN1 parameters must have requires_grad=False
        for name, param in nn1.named_parameters():
            assert not param.requires_grad, f"NN1 param '{name}' must have requires_grad=False"

        # All NN2 parameters must have requires_grad=False
        for name, param in nn2.named_parameters():
            assert not param.requires_grad, f"NN2 param '{name}' must have requires_grad=False"

        # All NN3 parameters must have requires_grad=True
        for name, param in nn3.named_parameters():
            assert param.requires_grad, f"NN3 param '{name}' must have requires_grad=True"

        # Eval vs Train mode
        assert not nn1.training
        assert not nn2.training
        assert nn3.training

    def test_zero_gradients_on_frozen_encoders(self):
        """
        Assert that after backward(), NN1 and NN2 gradients are None or 0,
        while NN3 gradients are non-zero.
        """
        device = torch.device("cpu")
        nn1 = NN1(embedding_dim=128).to(device)
        nn2 = NN2(embedding_dim=128).to(device)
        nn3 = NN3(in_features=128).to(device)

        freeze_encoders(nn1, nn2)
        unfreeze_regressor(nn3)

        criterion = MultiModalPoseLoss(alpha=1.0)
        optimizer = AdamW(nn3.parameters(), lr=1e-3)

        # Forward pass on virtual slice
        dummy_slice = torch.randn(2, 1, 227, 227, device=device)
        p_gt = torch.zeros(2, 9, device=device)
        p_gt[:, :3] = 0.1
        p_gt[:, 3] = 1.0
        p_gt[:, 7] = 1.0

        emb1 = nn1(dummy_slice)
        p_pred = nn3(emb1)

        loss = criterion(emb1=emb1, p_pred=p_pred, p_gt=p_gt)
        optimizer.zero_grad()
        loss.backward()

        # 1. NN1 gradients must be None (or all zeros)
        for name, param in nn1.named_parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0), f"NN1 param '{name}' has non-zero gradient: {param.grad}"

        # 2. NN2 gradients must be None (or all zeros)
        for name, param in nn2.named_parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0), f"NN2 param '{name}' has non-zero gradient: {param.grad}"

        # 3. NN3 gradients must exist and be non-zero
        nn3_grads_exist = False
        for name, param in nn3.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                nn3_grads_exist = True
                break
        assert nn3_grads_exist, "NN3 must have non-zero gradients for optimization!"

    def test_nn3_weights_update_while_encoders_remain_identical(self):
        """
        Verify NN3 weights change after optimizer.step() while NN1 and NN2 weights
        remain bit-exact identical.
        """
        device = torch.device("cpu")
        nn1 = NN1(embedding_dim=128).to(device)
        nn2 = NN2(embedding_dim=128).to(device)
        nn3 = NN3(in_features=128).to(device)

        freeze_encoders(nn1, nn2)
        unfreeze_regressor(nn3)

        # Snapshot weights before optimization
        nn1_before = {name: param.clone() for name, param in nn1.named_parameters()}
        nn2_before = {name: param.clone() for name, param in nn2.named_parameters()}
        nn3_before = {name: param.clone() for name, param in nn3.named_parameters()}

        criterion = MultiModalPoseLoss(alpha=1.0)
        optimizer = AdamW(nn3.parameters(), lr=1e-2)

        dummy_slice = torch.randn(2, 1, 227, 227, device=device)
        p_gt = torch.zeros(2, 9, device=device)
        p_gt[:, :3] = 0.5
        p_gt[:, 3] = 1.0
        p_gt[:, 7] = 1.0

        emb1 = nn1(dummy_slice)
        p_pred = nn3(emb1)
        loss = criterion(emb1=emb1, p_pred=p_pred, p_gt=p_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # NN1 weights must be 100% identical
        for name, param in nn1.named_parameters():
            assert torch.equal(param, nn1_before[name]), f"NN1 param '{name}' was modified!"

        # NN2 weights must be 100% identical
        for name, param in nn2.named_parameters():
            assert torch.equal(param, nn2_before[name]), f"NN2 param '{name}' was modified!"

        # NN3 weights must have updated
        nn3_changed = False
        for name, param in nn3.named_parameters():
            if not torch.equal(param, nn3_before[name]):
                nn3_changed = True
                break
        assert nn3_changed, "NN3 parameters should have updated after optimizer step!"

    def test_evaluate_patient_calibration_function(self):
        """Verify the validation evaluation function computes physical metrics."""
        device = torch.device("cpu")
        reslicer = VolumeReslicer(slice_size=(227, 227), rotation_mode="rotation_6d").to(device)
        nn1 = NN1(embedding_dim=128).to(device)
        nn3 = NN3(in_features=128).to(device)

        freeze_encoders(nn1, NN2(embedding_dim=128))
        unfreeze_regressor(nn3)

        vol_tensor = torch.randn(1, 1, 32, 32, 32, device=device)
        vol_shape = (32, 32, 32)
        spacing = (1.0, 1.0, 1.0)

        metrics = evaluate_patient_calibration(
            reslicer=reslicer,
            nn1=nn1,
            nn3=nn3,
            vol_tensor=vol_tensor,
            vol_shape=vol_shape,
            spacing=spacing,
            val_steps=2,
            batch_size=2,
            trans_bound_mm=20.0,
            rot_bound_deg=30.0,
            device=device,
        )

        assert "mte_mm" in metrics
        assert "mae_deg" in metrics
        assert isinstance(metrics["mte_mm"], float)
        assert isinstance(metrics["mae_deg"], float)
        assert metrics["mte_mm"] >= 0.0
        assert metrics["mae_deg"] >= 0.0

    def test_end_to_end_calibration_dry_run(self, tmp_path):
        """Verify calibrate_patient executes end-to-end in dry_run mode."""
        plot_path = tmp_path / "test_calibration_curves.png"
        ckpt_path = tmp_path / "test_calibrated_model.pth"

        history = calibrate_patient(
            case_name="TestPatient",
            data_path=None,  # will generate synthetic volume
            checkpoint_path=Path("non_existent_ckpt.pth"),
            epochs=2,
            steps_per_epoch=2,
            val_steps=2,
            batch_size=2,
            dry_run=True,
            output_plot=plot_path,
            save_checkpoint=ckpt_path,
            device_name="cpu",
            allow_synthetic=True,
        )

        assert "epoch" in history
        assert "train_loss" in history
        assert "val_mte_mm" in history
        assert "val_mae_deg" in history
        # Epochs: 0 (pre-calibration), 1, 2
        assert len(history["epoch"]) == 3
        assert len(history["val_mte_mm"]) == 3
        assert len(history["val_mae_deg"]) == 3
        assert plot_path.exists()


def test_reestimate_batchnorm_matches_target_statistics():
    """BatchNorm re-estimation sets the running statistics to those of the target data and keeps weights fixed."""
    from scripts.calibrate_patient import reestimate_batchnorm
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
    w_before = net[0].weight.detach().clone()
    data = torch.randn(64, 1, 8, 8) * 3.0 + 5.0
    assert reestimate_batchnorm(net, data, batch_size=16) == 1
    with torch.no_grad():
        feats = net[0](data)
    assert torch.allclose(net[1].running_mean, feats.mean(dim=(0, 2, 3)), atol=1e-4)
    assert torch.equal(net[0].weight, w_before)
    assert not net.training
