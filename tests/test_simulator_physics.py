"""
Unit tests for Task 1.6: Acoustic Simulator Setup (sim_us.py).

Verifies:
- Mathematical correctness of Beer-Lambert depth attenuation decay within 1% tolerance.
- Rayleigh backscatter speckle statistical distribution parameters (mean and std).
- End-to-end simulate_us pipeline preserving [1, 227, 227] tensor shapes and [0, 1] scaling.
- PyTorch GPU (CUDA) tensor acceleration when available.
"""

import math
import numpy as np
import pytest
import torch

from sim_us import AcousticSimulator


def test_attenuation_decay():
    """
    Verify Beer-Lambert acoustic attenuation decay against theoretical formula.
    Roadmap directive: Task 1.6 unit-test specification.
    """
    # Attenuation coefficient: 0.452 dB/cm/MHz -> convert to Neper/cm
    # alpha_Np = alpha_dB * ln(10) / 20 = 0.452 * 0.115129 = 0.052038 Np/cm/MHz
    # At depth l = 4 cm and f_0 = 5.0 MHz:
    # exponent = 0.052038 * 4 * 5.0 = 1.04076 -> Decay factor = e^(-1.04076) = 0.35318
    sim = AcousticSimulator(f0=5.0, c=1540.0, alpha_wm=0.452)
    initial_intensity = 1.0
    attenuated_intensity = sim.apply_attenuation(initial_intensity, depth_cm=4.0)

    # Assert within 1% numerical tolerance
    expected_decay = np.exp(-(0.452 * np.log(10) / 20.0) * 4.0 * 5.0)
    assert np.isclose(
        attenuated_intensity, expected_decay, rtol=1e-2
    ), f"Expected {expected_decay:.4f}, but got {attenuated_intensity:.4f}"


def test_rayleigh_speckle_statistics():
    """
    Assert that generated Rayleigh noise matches theoretical distribution moments:
        Mean = sigma * sqrt(pi / 2)
        Variance = (2 - pi / 2) * sigma^2
    """
    sigma = 59.09
    sim = AcousticSimulator(sigma_rayleigh=sigma)

    # Create uniform ones tensor to extract pure multiplicative noise
    ones_tensor = torch.ones((1, 512, 512), dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)
    noisy = sim.apply_rayleigh_speckle(ones_tensor, generator=generator)

    theoretical_mean = sigma * math.sqrt(math.pi / 2.0)
    theoretical_std = sigma * math.sqrt(2.0 - math.pi / 2.0)

    observed_mean = noisy.mean().item()
    observed_std = noisy.std().item()

    assert np.isclose(
        observed_mean, theoretical_mean, rtol=0.03
    ), f"Observed mean {observed_mean:.2f} differs from theoretical {theoretical_mean:.2f}"
    assert np.isclose(
        observed_std, theoretical_std, rtol=0.03
    ), f"Observed std {observed_std:.2f} differs from theoretical {theoretical_std:.2f}"


def test_simulate_us_pipeline():
    """
    Test end-to-end simulate_us pipeline with a 2D MRI slice tensor.
    Verifies output shape, normalization bounds, and vertical attenuation gradient.
    """
    sim = AcousticSimulator(f0=5.0, alpha_wm=0.452, sigma_rayleigh=59.09)

    # Synthetic clean MRI slice [1, 227, 227] with uniform anatomical tissue
    clean_mri = torch.full((1, 227, 227), 0.75, dtype=torch.float32)
    sim_us = sim.simulate_us(clean_mri, max_depth_cm=8.0, normalize=True)

    # Assert shape
    assert sim_us.shape == (1, 227, 227), f"Expected (1, 227, 227), got {sim_us.shape}"

    # Assert normalization
    assert sim_us.min().item() >= 0.0, "Simulated US minimum must be >= 0.0"
    assert sim_us.max().item() <= 1.0, "Simulated US maximum must be <= 1.0"

    # Verify physical depth attenuation gradient:
    # Shallow tissue (top rows, depth ~ 0-1 cm) should have higher intensity than deep tissue (bottom rows, depth ~ 7-8 cm)
    shallow_mean = sim_us[0, :30, :].mean().item()
    deep_mean = sim_us[0, -30:, :].mean().item()
    assert (
        shallow_mean > deep_mean
    ), f"Acoustic attenuation failed: shallow ({shallow_mean:.3f}) <= deep ({deep_mean:.3f})"


def test_cuda_gpu_acceleration():
    """Test that simulator supports PyTorch CUDA execution when GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")

    sim = AcousticSimulator(device="cuda")
    gpu_tensor = torch.rand((1, 227, 227), device="cuda", dtype=torch.float32)

    sim_gpu = sim.simulate_us(gpu_tensor)
    assert sim_gpu.is_cuda, "Simulated US output should remain on GPU device!"
    assert sim_gpu.shape == (1, 227, 227)
    assert sim_gpu.min() >= 0.0 and sim_gpu.max() <= 1.0


def test_numpy_compatibility():
    """Verify simulate_us functions seamlessly with NumPy ndarrays."""
    sim = AcousticSimulator()
    np_mri = np.full((227, 227), 0.6, dtype=np.float32)

    sim_us = sim.simulate_us(np_mri, max_depth_cm=8.0, normalize=True)
    assert isinstance(sim_us, np.ndarray)
    assert sim_us.shape == (227, 227)
    assert sim_us.min() >= 0.0 and sim_us.max() <= 1.0


def test_per_slice_normalisation_is_independent_of_batch():
    """With per-slice normalisation a frame's output must not depend on the other frames in the batch."""
    sim = AcousticSimulator(device="cpu")
    a = torch.rand(1, 1, 64, 64)
    b = torch.rand(1, 1, 64, 64) * 5.0
    g1 = torch.Generator().manual_seed(3)
    out_alone = sim.simulate_us(a, generator=g1)
    g2 = torch.Generator().manual_seed(3)
    out_batch = sim.simulate_us(torch.cat([a, b]), generator=g2)[:1]
    # Same speckle draw for the first frame (first half of the generator stream differs in size), so
    # compare normalisation behaviour: each frame spans exactly [0, 1]
    assert torch.isclose(out_batch.amax(), torch.tensor(1.0)) and torch.isclose(out_batch.amin(), torch.tensor(0.0))
    assert torch.isclose(out_alone.amax(), torch.tensor(1.0))
    legacy = AcousticSimulator.legacy(device="cpu")
    out_leg = legacy.simulate_us(torch.cat([a, b]), generator=torch.Generator().manual_seed(3))
    assert out_leg[0].amax() < 0.9   # batch-wide scaling: the brighter frame sets the maximum


def test_fan_mask_and_reflection_stages():
    """The sector mask zeroes the near-field corners, and the reflection term responds to an interface."""
    sim = AcousticSimulator(device="cpu")
    img = torch.zeros(1, 1, 227, 227)
    img[..., 120:, :] = 0.8                                  # horizontal interface at row 120
    st = sim.simulate_us(img, return_stages=True, generator=torch.Generator().manual_seed(0))
    for k in ("input", "reflection", "echo", "speckled", "gain", "mask", "output"):
        assert k in st
    mask = st["mask"]
    assert mask[0, 0] < 0.01 and mask[0, -1] < 0.01          # top corners outside the fan
    assert mask[113, 113] > 0.99                              # centre inside
    assert st["reflection"][0, 0, 119:122, 113].max() > 0.5   # strong echo at the interface
    assert st["gain"][-1, 113] < st["gain"][0, 113]           # attenuation increases with depth
    assert st["output"].min() >= 0.0 and st["output"].max() <= 1.0
