"""
Unit tests for Task 2.2: Supervised Multi-Positive InfoNCE Loss.

Verifies:
- Loss positivity and bounded magnitude on spatially aligned slices (Roadmap directive).
- Gradient backpropagation with non-zero, finite gradients.
- Physical spatial Euclidean distance thresholding (delta = 1.5 mm).
- Multimodal paired cross-modal interface (features_us, coordinates, features_mr).
- CUDA GPU acceleration when available.
"""

import pytest
import torch

from losses import MultiPositiveInfoNCELoss


def test_loss_on_identical_slices():
    """
    Assert loss value properties on identical feature pairs within distance threshold delta=1.5mm.
    Roadmap directive: Task 2.2 unit-test specification.
    """
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.07, distance_threshold=1.5)

    # 2 pairs of identical features
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    # Coordinates aligned in pairs within delta = 1.5mm
    coordinates = torch.tensor(
        [[10.0, 0.0, 0.0], [10.5, 0.0, 0.0], [50.0, 0.0, 0.0], [50.2, 0.0, 0.0]]
    )

    loss_val = loss_fn(features, coordinates)
    assert loss_val > 0.0, "Loss must be positive!"
    assert loss_val.item() < 3.0, "Loss value too high for perfectly aligned pairs!"


def test_gradient_backpropagation():
    """Verify that gradients backpropagate smoothly with no NaNs."""
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.07, distance_threshold=1.5)

    features = torch.randn(8, 128, requires_grad=True)
    # 4 pairs of adjacent physical slices
    coordinates = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.8, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [31.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [40.3, 0.0, 0.0],
        ]
    )

    loss = loss_fn(features, coordinates)
    assert loss.requires_grad
    loss.backward()

    assert features.grad is not None, "Gradients were not computed!"
    assert not torch.isnan(features.grad).any(), "Found NaN values in feature gradients!"
    assert features.grad.norm().item() > 0.0, "Gradients are all zero!"


def test_distance_threshold_sensitivity():
    """
    Assert that slices within delta=1.5mm have lower loss than when spatial
    distances exceed delta and are treated as contrastive negatives.
    """
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.07, distance_threshold=1.5)

    # Identical features across all pairs
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
    )

    # Case 1: Pairs within delta (10.0 and 10.5 -> dist 0.5 <= 1.5)
    coords_near = torch.tensor(
        [[10.0, 0.0, 0.0], [10.5, 0.0, 0.0], [50.0, 0.0, 0.0], [50.5, 0.0, 0.0]]
    )
    loss_near = loss_fn(features, coords_near)

    # Case 2: Inverted alignment (dist 40mm between identical features)
    coords_far = torch.tensor(
        [[10.0, 0.0, 0.0], [50.0, 0.0, 0.0], [10.5, 0.0, 0.0], [50.5, 0.0, 0.0]]
    )
    loss_far = loss_fn(features, coords_far)

    assert (
        loss_near < loss_far
    ), f"Spatially aligned loss ({loss_near.item():.4f}) should be lower than misaligned ({loss_far.item():.4f})"


def test_cross_modal_forward_interface():
    """Verify multimodal call signature: forward(features_us, coordinates, features_mr)."""
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.07, distance_threshold=1.5)

    batch_size = 4
    z_us = torch.randn(batch_size, 128)
    z_mr = torch.randn(batch_size, 128)
    coords = torch.tensor(
        [[0.0, 0.0, float(i * 1.0)] for i in range(batch_size)]
    )

    loss = loss_fn(z_us, coords, features_mr=z_mr)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0.0


def test_cuda_device_compatibility():
    """Verify loss computation directly on CUDA GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on this machine.")

    loss_fn = MultiPositiveInfoNCELoss().cuda()
    features = torch.randn(6, 128, device="cuda", requires_grad=True)
    coords = torch.tensor(
        [[0.0, 0.0, float(i * 0.8)] for i in range(6)], device="cuda"
    )

    loss = loss_fn(features, coords)
    assert loss.is_cuda
    loss.backward()
    assert features.grad is not None and features.grad.is_cuda
