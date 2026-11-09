"""
Volume loading must fail loudly on real data problems and never silently substitute a synthetic phantom.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_pose_pipeline import load_or_create_volume  # noqa: E402

CPU = torch.device("cpu")


def test_missing_volume_raises_by_default(tmp_path):
    """A path with no MRI volume raises FileNotFoundError unless a phantom is explicitly allowed."""
    with pytest.raises(FileNotFoundError):
        load_or_create_volume(tmp_path / "Case99-FLAIR.nii.gz", CPU)
    with pytest.raises(FileNotFoundError):
        load_or_create_volume(tmp_path, CPU)          # directory without any volume file


def test_corrupt_volume_raises_even_when_synthetic_allowed(tmp_path):
    """An existing but unreadable NIfTI file is always an error, never replaced by a phantom."""
    bad = tmp_path / "Case99-FLAIR.nii.gz"
    bad.write_bytes(b"this is not a NIfTI volume")
    with pytest.raises(Exception):
        load_or_create_volume(bad, CPU)
    with pytest.raises(Exception):
        load_or_create_volume(bad, CPU, allow_synthetic=True)


def test_synthetic_phantom_only_when_requested(tmp_path):
    vol, shape, spacing = load_or_create_volume(tmp_path / "absent.nii.gz", CPU, allow_synthetic=True)
    assert shape == (128, 128, 128) and spacing == (1.0, 1.0, 1.0)
    assert vol.shape == (1, 1, 128, 128, 128)


def test_valid_volume_loads(tmp_path):
    data = np.random.default_rng(0).random((8, 10, 12)).astype(np.float32)
    path = tmp_path / "Case99-FLAIR.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0])), str(path))
    vol, shape, spacing = load_or_create_volume(path, CPU, canonical=True, plane="axial")
    assert shape == (12, 10, 8)                       # axial plane permutes RAS to (S, A, R)
    assert spacing == (2.0, 1.0, 1.0)
    assert 0.0 <= float(vol.min()) and float(vol.max()) <= 1.0
