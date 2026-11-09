"""
Unit tests for Task 1.5: Blank-Slice Gating System.

Verifies the non-trainable binary pre-filtering gate to eliminate low-signal
acoustic background slices (<500 active pixels) and stabilize training gradients.
"""

import numpy as np
import pytest
import torch

from dataset import BlankSliceGate


def test_blank_slice_gating_operator():
    """Verify BlankSliceGate instantiated as a callable operator (Roadmap directive)."""
    gate = BlankSliceGate(active_pixel_val=0.1, min_pixels=500)

    # Test 1: Complete blank image (all zeros)
    blank_img = torch.zeros((1, 227, 227))
    assert gate(blank_img) is False, "Failed to gate a complete blank slice!"

    # Test 2: Low-signal boundary image (300 active pixels)
    sparse_img = torch.zeros((1, 227, 227))
    sparse_img[0, 10:20, 10:40] = 0.5  # 10 * 30 = 300 active pixels
    assert gate(sparse_img) is False, "Failed to gate sparse slice with 300 active pixels!"

    # Test 3: Valid anatomical slice (1000 active pixels)
    valid_img = torch.zeros((1, 227, 227))
    valid_img[0, 10:30, 10:60] = 0.8  # 20 * 50 = 1000 active pixels
    assert gate(valid_img) is True, "Incorrectly gated a valid anatomical slice!"


def test_blank_slice_gating_functional_call():
    """Verify BlankSliceGate callable directly as a function."""
    blank_img = torch.zeros((1, 227, 227))
    assert BlankSliceGate(blank_img, min_pixels=500) is False

    valid_img = torch.zeros((1, 227, 227))
    valid_img[0, 0:25, 0:25] = 0.5  # 25 * 25 = 625 active pixels
    assert BlankSliceGate(valid_img, min_pixels=500) is True


def test_blank_slice_gating_numpy_input():
    """Verify BlankSliceGate handles NumPy ndarrays seamlessly."""
    gate = BlankSliceGate(active_pixel_val=0.1, min_pixels=500)

    np_blank = np.zeros((227, 227), dtype=np.float32)
    assert gate(np_blank) is False

    np_valid = np.zeros((227, 227), dtype=np.float32)
    np_valid[0:30, 0:30] = 0.9  # 900 active pixels
    assert gate(np_valid) is True
