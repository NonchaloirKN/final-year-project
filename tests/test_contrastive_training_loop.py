"""
Unit tests for Siamese Contrastive Pre-Training Loop & Architecture (Task 2.3).

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Verifies:
1. Single forward-backward pass and optimization step decreases loss without NaN gradients.
2. Leave-One-Subject-Out (LOSO) data partitioning guarantees disjoint train/val splits.
3. Model checkpoint serialization and reload produces identical parameters and predictions.
4. CosineAnnealingLR scheduler smoothly decreases learning rate over epochs.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from models import DualEncoderResNet18
from losses import MultiPositiveInfoNCELoss
from scripts.train_contrastive import (
    SyntheticPhantomDataset,
    collate_contrastive_batch,
    get_loso_splits,
    train_one_epoch,
)


def test_single_step_optimization_loss_decrease():
    """
    Assert that a single forward-backward pass and optimization step on a synthetic mini-batch
    produces non-NaN gradients and decreases the loss on that batch.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = DualEncoderResNet18(embedding_dim=128, siamese=True).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = MultiPositiveInfoNCELoss(temperature=0.07, distance_threshold=1.5)

    # Mini-batch of 4 paired slices
    us_batch = torch.randn(4, 1, 227, 227, device=device)
    mri_batch = torch.randn(4, 1, 227, 227, device=device)
    # Coordinates: first 2 within delta=1.5mm, second 2 within delta=1.5mm
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 10.0], [0.0, 0.0, 11.0]],
        device=device,
    )

    # Step 1: Initial evaluation & backward pass
    optimizer.zero_grad()
    z_us1, z_mr1 = model(us_batch, mri_batch)
    loss1 = criterion(z_us1, coords, z_mr1)
    assert not torch.isnan(loss1), "Initial loss is NaN!"
    assert not torch.isinf(loss1), "Initial loss is Inf!"
    assert loss1.item() > 0.0, "Loss must be positive!"

    loss1.backward()

    # Verify no NaN or Inf in gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}!"
            assert not torch.isinf(param.grad).any(), f"Inf gradient detected in {name}!"

    optimizer.step()

    # Step 2: Loss on identical mini-batch must decrease
    optimizer.zero_grad()
    z_us2, z_mr2 = model(us_batch, mri_batch)
    loss2 = criterion(z_us2, coords, z_mr2)
    assert loss2.item() < loss1.item(), (
        f"Loss failed to decrease after optimization step! Step 1: {loss1.item():.4f}, Step 2: {loss2.item():.4f}"
    )


def test_loso_partitioning_disjointness(tmp_path: Path):
    """
    Assert that Leave-One-Subject-Out (LOSO) data partitioning strictly isolates
    the validation subject with zero identity leakage into the training split.
    """
    # Create mock dataset directory structure: Case1 to Case23
    for i in range(1, 24):
        case_dir = tmp_path / f"Case{i}"
        case_dir.mkdir(parents=True, exist_ok=True)

    val_case = "Case23"
    train_cases, val_cases = get_loso_splits(tmp_path, val_case=val_case)

    # Asserts
    assert val_case in val_cases, f"Validation case {val_case} not in val_cases!"
    assert len(val_cases) == 1, "Validation split must contain exactly one subject!"
    assert val_case not in train_cases, f"Data leakage: {val_case} found in training cases!"
    assert len(train_cases) == 22, f"Expected 22 training cases, got {len(train_cases)}"
    assert set(train_cases).isdisjoint(set(val_cases)), "Train and Val case sets must be disjoint!"


def test_checkpoint_serialization_and_reload(tmp_path: Path):
    """
    Assert that model checkpoints can be serialized to disk and loaded into a new
    model instance, producing bitwise-identical parameters and identical latent embeddings.
    """
    torch.manual_seed(123)
    model1 = DualEncoderResNet18(embedding_dim=128, siamese=True)
    model1.eval()

    checkpoint_file = tmp_path / "test_best_model.pth"
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": model1.state_dict(),
            "val_loss": 1.2345,
            "val_cosine": 0.8765,
            "embedding_dim": 128,
        },
        str(checkpoint_file),
    )
    assert checkpoint_file.exists(), "Checkpoint file was not written to disk!"

    # Instantiate fresh model and load checkpoint
    model2 = DualEncoderResNet18(embedding_dim=128, siamese=True)
    checkpoint = torch.load(str(checkpoint_file), weights_only=True)
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()

    # Check parameter equality
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Model parameters differ after checkpoint reload!"

    # Check output equality
    dummy_us = torch.randn(2, 1, 227, 227)
    dummy_mr = torch.randn(2, 1, 227, 227)
    with torch.no_grad():
        z_us1, z_mr1 = model1(dummy_us, dummy_mr)
        z_us2, z_mr2 = model2(dummy_us, dummy_mr)

    assert torch.allclose(z_us1, z_us2, atol=1e-6), "US embeddings differ after checkpoint reload!"
    assert torch.allclose(z_mr1, z_mr2, atol=1e-6), "MR embeddings differ after checkpoint reload!"


def test_cosine_annealing_scheduler_progression():
    """
    Assert that CosineAnnealingLR decreases learning rate smoothly across training epochs.
    """
    model = DualEncoderResNet18(embedding_dim=128, siamese=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    lrs = []
    for _ in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Initial lr should be close to 1e-3
    assert abs(lrs[0] - 1e-3) < 1e-6
    # Final lr should be lower than initial lr
    assert lrs[-1] < lrs[0]
    # Learning rate should decrease monotonically in first half of cosine period
    for i in range(len(lrs) - 1):
        assert lrs[i + 1] <= lrs[i] + 1e-9, f"LR did not decrease at epoch {i}: {lrs[i]} -> {lrs[i+1]}"


def test_synthetic_phantom_dataset():
    """
    Verify SyntheticPhantomDataset properties and batch collation with 9D pose tensors.
    """
    ds = SyntheticPhantomDataset(num_samples=8, size=227)
    assert len(ds) == 8

    us, mri, pose = ds[0]
    assert us.shape == (1, 227, 227)
    assert mri.shape == (1, 227, 227)
    assert isinstance(pose, torch.Tensor)
    assert pose.shape == (9,)
    assert pose[2] == 50.0  # z_pos = 50.0 + 0 * 2.0

    batch = [ds[i] for i in range(4)]
    us_b, mr_b, coords_b = collate_contrastive_batch(batch)
    assert us_b.shape == (4, 1, 227, 227)
    assert mr_b.shape == (4, 1, 227, 227)
    assert coords_b.shape == (4, 3)
    # Check that z coordinates were properly collated from index 2
    for i in range(4):
        expected_z = 50.0 + i * 2.0
        assert coords_b[i, 2].item() == expected_z
        assert coords_b[i, 0].item() == 0.0
        assert coords_b[i, 1].item() == 0.0


def test_collate_contrastive_batch_heterogeneous_formats():
    """
    Verify collate_contrastive_batch robustly handles 9D tensors, 3D tensors,
    legacy dictionaries, and scalars without throwing AttributeError.
    """
    dummy_us = torch.zeros(1, 227, 227)
    dummy_mr = torch.zeros(1, 227, 227)

    # 1. Continuous 9D pose tensor: [tx=0, ty=0, tz=100.0, r1..r6]
    pose_9d = torch.tensor([0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # 2. 3D translation tensor: [0.0, 0.0, 102.0]
    pose_3d = torch.tensor([0.0, 0.0, 102.0])
    # 3. Legacy 3D tensor: [104.0, 0.0, 0.0]
    pose_legacy_3d = torch.tensor([104.0, 0.0, 0.0])
    # 4. Legacy dictionary: {"z_coord": 106.0}
    pose_dict = {"z_coord": 106.0}
    # 5. Scalar float: 108.0
    pose_scalar = 108.0

    batch = [
        (dummy_us, dummy_mr, pose_9d),
        (dummy_us, dummy_mr, pose_3d),
        (dummy_us, dummy_mr, pose_legacy_3d),
        (dummy_us, dummy_mr, pose_dict),
        (dummy_us, dummy_mr, pose_scalar),
    ]

    us_b, mr_b, coords_b = collate_contrastive_batch(batch)
    assert us_b.shape == (5, 1, 227, 227)
    assert mr_b.shape == (5, 1, 227, 227)
    assert coords_b.shape == (5, 3)

    expected_z_values = [100.0, 102.0, 104.0, 106.0, 108.0]
    for i, exp_z in enumerate(expected_z_values):
        assert coords_b[i, 2].item() == exp_z, f"Item {i} failed: expected {exp_z}, got {coords_b[i, 2].item()}"

