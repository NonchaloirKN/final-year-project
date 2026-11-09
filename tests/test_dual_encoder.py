"""
Unit tests for Task 2.1: Dual-Encoder ResNet18 Architecture.

Verifies:
- Input/output tensor dimensions ([Batch, 1, 227, 227] -> [Batch, 128]).
- L2 normalization of output embeddings.
- Weight sharing properties under Siamese mode (siamese=True).
- Weight decoupling under non-Siamese mode (siamese=False) and via decouple_weights().
- Gradient backpropagation through both encoders.
- CUDA device compatibility when GPU is present.
"""

import pytest
import torch

from models import DualEncoderResNet18


def test_encoder_shapes_and_weight_sharing():
    """
    Assert tensor shapes and Siamese weight sharing as specified in Roadmap Task 2.1.
    """
    model = DualEncoderResNet18(embedding_dim=128, siamese=True)
    us_input = torch.randn(2, 1, 227, 227)
    mri_input = torch.randn(2, 1, 227, 227)

    z_us, z_mr = model(us_input, mri_input)

    assert z_us.shape == (2, 128), f"Expected shape (2, 128), got {z_us.shape}"
    assert z_mr.shape == (2, 128), f"Expected shape (2, 128), got {z_mr.shape}"

    # Test weight sharing: weights of ResNet backbones must be identical
    param_us = list(model.encoder_us.parameters())[0]
    param_mr = list(model.encoder_mr.parameters())[0]
    assert torch.equal(param_us, param_mr), "Siamese weight sharing failed!"


def test_embedding_l2_normalization():
    """Verify that output embeddings have unit L2 norm."""
    model = DualEncoderResNet18(embedding_dim=128, siamese=True)
    us_input = torch.randn(4, 1, 227, 227)
    mri_input = torch.randn(4, 1, 227, 227)

    z_us, z_mr = model(us_input, mri_input)

    norm_us = torch.norm(z_us, p=2, dim=1)
    norm_mr = torch.norm(z_mr, p=2, dim=1)

    assert torch.allclose(norm_us, torch.ones_like(norm_us), atol=1e-5), "z_us is not L2 normalized!"
    assert torch.allclose(norm_mr, torch.ones_like(norm_mr), atol=1e-5), "z_mr is not L2 normalized!"


def test_encoder_decoupling():
    """Verify that weights can be initialized decoupled or dynamically decoupled."""
    # Decoupled at initialization
    model_decoupled = DualEncoderResNet18(embedding_dim=128, siamese=False)
    param_us = list(model_decoupled.encoder_us.parameters())[0]
    param_mr = list(model_decoupled.encoder_mr.parameters())[0]
    assert not torch.equal(param_us, param_mr), "Decoupled encoders should have independent parameters!"

    # Dynamic decoupling from Siamese
    model_siamese = DualEncoderResNet18(embedding_dim=128, siamese=True)
    assert model_siamese.siamese is True
    model_siamese.decouple_weights()
    assert model_siamese.siamese is False
    # Parameter values initially match, but object references must differ
    param_us_dyn = list(model_siamese.encoder_us.parameters())[0]
    param_mr_dyn = list(model_siamese.encoder_mr.parameters())[0]
    assert param_us_dyn is not param_mr_dyn, "Encoders must be distinct objects after decoupling!"


def test_dual_encoder_gradients():
    """Verify gradient backpropagation through both encoder branches."""
    model = DualEncoderResNet18(embedding_dim=128, siamese=True)
    us_input = torch.randn(2, 1, 227, 227)
    mri_input = torch.randn(2, 1, 227, 227)

    z_us, z_mr = model(us_input, mri_input)
    dummy_loss = (z_us * z_mr).sum()
    dummy_loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"


def test_dual_encoder_cuda():
    """Verify execution on CUDA GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable on this machine.")

    model = DualEncoderResNet18(embedding_dim=128, siamese=True).cuda()
    us_input = torch.randn(2, 1, 227, 227, device="cuda")
    mri_input = torch.randn(2, 1, 227, 227, device="cuda")

    z_us, z_mr = model(us_input, mri_input)
    assert z_us.is_cuda and z_mr.is_cuda
    assert z_us.shape == (2, 128)
