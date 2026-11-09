"""
Unit tests for Task 1.4: Dynamic Bounding Box Jitter.

Verifies bounding box volume preservation and translation offset bounds
within +/-5 to +/-10 voxels along X, Y, and Z axes to prevent edge shortcut overfitting.
"""

import numpy as np
import pytest

from dataset import apply_bbox_jitter


def test_bbox_jitter_offset():
    """Verify jitter offset boundaries and volume preservation as specified in Roadmap."""
    base_roi = [100, 100, 100, 150, 150, 150]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    np.random.seed(42)  # For reproducible testing

    jittered_roi, offset = apply_bbox_jitter(base_roi, min_jitter=5, max_jitter=10)

    # Verify bounding box volume is preserved (50x50x50)
    vol_original = (
        (base_roi[3] - base_roi[0])
        * (base_roi[4] - base_roi[1])
        * (base_roi[5] - base_roi[2])
    )
    vol_jittered = (
        (jittered_roi[3] - jittered_roi[0])
        * (jittered_roi[4] - jittered_roi[1])
        * (jittered_roi[5] - jittered_roi[2])
    )

    assert vol_original == vol_jittered, "Bounding box volume was not preserved!"

    # Verify jitter falls within the specified bounds
    for d in offset:
        assert 5 <= abs(d) <= 10, f"Jitter offset {d} out of bounds [5, 10]!"


def test_bbox_jitter_statistical_distribution():
    """Assert offset bounds across 100 random draws."""
    base_roi = [50, 50, 50, 100, 100, 100]

    for _ in range(100):
        jittered, offsets = apply_bbox_jitter(base_roi, min_jitter=5, max_jitter=10)
        assert len(offsets) == 3

        # Check dimension lengths are preserved
        assert (jittered[3] - jittered[0]) == (base_roi[3] - base_roi[0])
        assert (jittered[4] - jittered[1]) == (base_roi[4] - base_roi[1])
        assert (jittered[5] - jittered[2]) == (base_roi[5] - base_roi[2])

        for d in offsets:
            assert 5 <= abs(d) <= 10, f"Jitter magnitude {abs(d)} outside [5, 10]!"


def test_bbox_jitter_invalid_input():
    """Assert exception on malformed ROI."""
    with pytest.raises(ValueError):
        apply_bbox_jitter([10, 20, 30])  # Only 3 elements
