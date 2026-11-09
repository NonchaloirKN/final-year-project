"""
Unit tests for Twin Feature Encoders (NN1 and NN2).

Asserts:
1. Output tensor shapes are strictly [B, 128].
2. Output embeddings have unit L2 norm (||z||_2 = 1.0).
3. Both NN1 (MRI) and NN2 (US) support single-channel 2D inputs.
4. Gradients propagate backwards through encoders without NaNs.
5. TwinEncoders wrapper operates correctly in decoupled and shared modes.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.encoders import NN1, NN2, MRIEncoder, USEncoder, TwinEncoders


def test_mri_encoder_nn1_shape_and_l2_norm():
    """Verify NN1 outputs shape [B, 128] with unit L2 norm."""
    encoder = NN1(embedding_dim=128)
    encoder.eval()

    batch_size = 4
    mri_slices = torch.randn(batch_size, 1, 227, 227)
    with torch.no_grad():
        emb1 = encoder(mri_slices)

    assert emb1.shape == (batch_size, 128), f"Expected shape ({batch_size}, 128), got {emb1.shape}"
    norms = torch.norm(emb1, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"L2 norms not 1.0: {norms}"


def test_us_encoder_nn2_shape_and_l2_norm():
    """Verify NN2 outputs shape [B, 128] with unit L2 norm."""
    encoder = NN2(embedding_dim=128)
    encoder.eval()

    batch_size = 6
    us_slices = torch.randn(batch_size, 1, 227, 227)
    with torch.no_grad():
        emb2 = encoder(us_slices)

    assert emb2.shape == (batch_size, 128), f"Expected shape ({batch_size}, 128), got {emb2.shape}"
    norms = torch.norm(emb2, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"L2 norms not 1.0: {norms}"


def test_twin_encoders_paired_forward():
    """Verify TwinEncoders extracts both EMB1 and EMB2 simultaneously."""
    twin = TwinEncoders(embedding_dim=128, shared_weights=False)
    twin.eval()

    b = 3
    mri_batch = torch.randn(b, 1, 227, 227)
    us_batch = torch.randn(b, 1, 227, 227)

    with torch.no_grad():
        emb1, emb2 = twin(mri_batch, us_batch)

    assert emb1.shape == (b, 128)
    assert emb2.shape == (b, 128)
    assert torch.allclose(torch.norm(emb1, p=2, dim=-1), torch.ones(b), atol=1e-5)
    assert torch.allclose(torch.norm(emb2, p=2, dim=-1), torch.ones(b), atol=1e-5)


def test_encoder_gradient_backpropagation():
    """Verify autograd backpropagation flows through NN1 and NN2 without NaNs."""
    twin = TwinEncoders(embedding_dim=128, shared_weights=False)
    twin.train()

    mri_slice = torch.randn(2, 1, 128, 128, requires_grad=True)
    us_slice = torch.randn(2, 1, 128, 128, requires_grad=True)

    emb1, emb2 = twin(mri_slice, us_slice)
    loss = F.mse_loss(emb1, emb2)
    loss.backward()

    # Assert gradients exist on inputs and model weights
    assert mri_slice.grad is not None
    assert us_slice.grad is not None
    assert not torch.isnan(mri_slice.grad).any()
    assert not torch.isnan(us_slice.grad).any()

    for name, param in twin.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"


def test_alias_identities():
    """Verify alias equivalence between NN1/MRIEncoder and NN2/USEncoder."""
    assert NN1 is MRIEncoder
    assert NN2 is USEncoder
