"""
Unit tests for Task 1.3: Custom RESECTDataset Class with Normalisation.

Verifies tensor types, spatial output dimensions ([1, 227, 227]), and
modality-specific intensity normalization (min-max for US, 99th-percentile clipping for MRI).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from dataset import RESECTDataset, normalize_mri, normalize_us

RESECT_DIR = Path("data/resect_base/RESECT/NIFTI")
requires_resect = pytest.mark.skipif(not RESECT_DIR.is_dir(), reason="RESECT data not available")


def test_normalize_us():
    """Test min-max intensity normalization for ultrasound images."""
    # Test typical US slice distribution
    raw = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    norm = normalize_us(raw)
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert np.isclose(norm[0, 1], 50.0 / 200.0)

    # Test uniform/flat image (should return all zeros, no division by zero)
    flat = np.ones((10, 10), dtype=np.float32) * 42.0
    norm_flat = normalize_us(flat)
    assert np.all(norm_flat == 0.0)


def test_normalize_mri():
    """Test 99th-percentile clipping and normalization for MRI slices."""
    # Create array with a severe hyperintense spike (fat artifact)
    raw = np.linspace(0.0, 100.0, 1000, dtype=np.float32)
    raw[-1] = 10000.0  # Extreme outlier

    norm = normalize_mri(raw, percentile=99.0)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    # The extreme outlier should be clipped to 1.0
    assert norm[-1] == 1.0


@requires_resect
def test_dataset_output_properties(tmp_path: Path):
    """
    Assert that RESECTDataset outputs PyTorch tensors of size [1, 227, 227]
    with values normalized strictly to [0.0, 1.0].
    """
    csv_file = tmp_path / "metadata.csv"

    # Limit to Case1 for rapid unit-test verification
    dataset = RESECTDataset(
        data_dir="data/resect_base/RESECT/NIFTI",
        csv_file=csv_file,
        resize=(227, 227),
        cases=["Case1"],
        sample_stride=10,
    )

    assert len(dataset) > 0, "Dataset contains zero samples for Case1!"

    us_img, mri_img, pose = dataset[0]

    # Assert type
    assert isinstance(us_img, torch.Tensor), "US image must be a PyTorch Tensor!"
    assert isinstance(mri_img, torch.Tensor), "MRI image must be a PyTorch Tensor!"
    assert isinstance(pose, torch.Tensor), "Pose must be a PyTorch Tensor!"

    # Assert shape [Channels, Height, Width]
    assert us_img.shape == (1, 227, 227), f"Expected shape (1, 227, 227), got {us_img.shape}"
    assert mri_img.shape == (1, 227, 227), f"Expected shape (1, 227, 227), got {mri_img.shape}"

    # Assert normalization
    assert us_img.min() >= 0.0 and us_img.max() <= 1.0, f"US normalization failed: {us_img.min()}, {us_img.max()}"
    assert mri_img.min() >= 0.0 and mri_img.max() <= 1.0, f"MRI normalization failed: {mri_img.min()}, {mri_img.max()}"


@requires_resect
def test_dataset_csv_reload(tmp_path: Path):
    """Verify metadata persistence and re-loading from CSV."""
    csv_file = tmp_path / "cached_metadata.csv"

    # Build and save
    ds1 = RESECTDataset(
        data_dir="data/resect_base/RESECT/NIFTI",
        csv_file=csv_file,
        cases=["Case1"],
        sample_stride=10,
    )
    assert csv_file.exists(), "Metadata CSV was not created!"

    # Reload from CSV
    ds2 = RESECTDataset(
        csv_file=csv_file,
        cases=["Case1"],
    )
    assert len(ds1) == len(ds2), f"Reloaded count {len(ds2)} != original {len(ds1)}"


def test_resect_patient_cases_canonical_cohort():
    """Verify that RESECT_PATIENT_CASES contains exactly 23 cases ending at Case27."""
    from dataset import RESECT_PATIENT_CASES

    assert len(RESECT_PATIENT_CASES) == 23, f"Expected 23 patient cases, got {len(RESECT_PATIENT_CASES)}"
    assert RESECT_PATIENT_CASES[0] == "Case1", f"First case must be Case1, got {RESECT_PATIENT_CASES[0]}"
    assert RESECT_PATIENT_CASES[-1] == "Case27", f"23rd case must be Case27, got {RESECT_PATIENT_CASES[-1]}"

    # Omitted case IDs in the RESECT dataset
    skipped_cases = ["Case9", "Case10", "Case20", "Case22"]
    for sc in skipped_cases:
        assert sc not in RESECT_PATIENT_CASES, f"Skipped {sc} should NOT be in RESECT cohort!"

    # Monotonically increasing numerical IDs
    ids = [int(c.replace("Case", "")) for c in RESECT_PATIENT_CASES]
    assert ids == sorted(ids), "Patient cases must be sorted in strictly increasing numerical order!"


def test_loso_splits_with_skipped_case_numbers(tmp_path: Path):
    """
    Assert that get_loso_splits correctly handles the non-contiguous RESECT cohort
    and properly withholds the final case (Case27) or any specified case (Case23).
    """
    from dataset import RESECT_PATIENT_CASES
    from scripts.train_contrastive import get_loso_splits

    # Create mock cohort with realistic non-contiguous Case1..Case27 directories
    for c in RESECT_PATIENT_CASES:
        (tmp_path / c).mkdir(parents=True, exist_ok=True)

    # Test 1: Withholding Case27 (the 23rd case)
    train_27, val_27 = get_loso_splits(tmp_path, val_case="Case27")
    assert val_27 == ["Case27"]
    assert len(train_27) == 22
    assert "Case27" not in train_27
    assert set(train_27).isdisjoint(set(val_27))

    # Test 2: Withholding Case23
    train_23, val_23 = get_loso_splits(tmp_path, val_case="Case23")
    assert val_23 == ["Case23"]
    assert len(train_23) == 22
    assert "Case23" not in train_23
    assert set(train_23).isdisjoint(set(val_23))

    # Test 3: Natural sort order verification (Case2 must precede Case11)
    assert train_27[1] == "Case2", f"Natural sorting failed: expected Case2 at index 1, got {train_27[1]}"
