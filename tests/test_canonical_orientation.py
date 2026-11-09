"""
Unit tests for canonical (RAS+) volume orientation (Sprint 8, Task S8.7c).

RESECT stores most T2-FLAIR volumes as (R, A, S) arrays but Case 4 as (P, I, R) and Case 11 as (A, S, R).
load_canonical_nifti must reorient every volume to RAS+ without resampling, preserving each voxel's world position.
"""

import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.utils.dataset import CANONICAL_AXCODES, load_canonical_nifti


def test_pir_volume_is_reoriented_to_ras(tmp_path):
    """A synthetic (P, I, R) volume is permuted and flipped to RAS+, and a marked voxel keeps its world position."""
    data = np.zeros((4, 5, 6), dtype=np.float32)
    data[1, 3, 2] = 7.0
    # Columns of the affine: axis0 -> -y (posterior), axis1 -> -z (inferior), axis2 -> +x (right)
    affine = np.array([[0, 0, 1.5, -10], [-1.0, 0, 0, 20], [0, -2.0, 0, 30], [0, 0, 0, 1]], dtype=float)
    path = tmp_path / "pir.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    assert nib.aff2axcodes(affine) == ("P", "I", "R")

    canon, zooms, original = load_canonical_nifti(path)
    assert original == ("P", "I", "R")
    assert canon.shape == (6, 4, 5)
    assert zooms == pytest.approx((1.5, 1.0, 2.0))

    world_before = affine @ np.array([1, 3, 2, 1.0])
    canon_img = nib.as_closest_canonical(nib.load(str(path)))
    idx = np.argwhere(canon == 7.0)[0]
    world_after = canon_img.affine @ np.append(idx, 1.0)
    assert np.allclose(world_before, world_after)
    assert nib.aff2axcodes(canon_img.affine) == CANONICAL_AXCODES


FLAIRS = sorted(glob.glob("data/resect_base/RESECT/NIFTI/Case*/MRI/*FLAIR*.nii.gz"))


@pytest.mark.skipif(not FLAIRS, reason="RESECT data not available")
def test_all_resect_volumes_share_ras_orientation():
    """Every RESECT FLAIR volume loads as RAS+, so array axes mean the same anatomy for all patients."""
    shapes = {}
    for f in FLAIRS:
        img = nib.as_closest_canonical(nib.load(f))
        assert tuple(nib.aff2axcodes(img.affine)) == CANONICAL_AXCODES, f
        shapes[Path(f).name] = img.shape
    for name in ("Case4-FLAIR.nii.gz", "Case11-FLAIR.nii.gz"):
        if name in shapes:
            assert shapes[name] == (192, 256, 256), (name, shapes[name])


def test_axial_plane_puts_superior_on_axis_zero(tmp_path):
    """plane='axial' permutes RAS to (S, A, R): axis 0 is inferior-superior, and spacing follows the axes."""
    data = np.zeros((4, 5, 6), dtype=np.float32)
    data[1, 2, 5] = 3.0
    affine = np.diag([1.0, 2.0, 3.0, 1.0])      # already RAS with distinct spacings
    path = tmp_path / "ras.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    sag, z_sag, _ = load_canonical_nifti(path, plane="sagittal")
    ax, z_ax, _ = load_canonical_nifti(path, plane="axial")
    assert sag.shape == (4, 5, 6) and ax.shape == (6, 5, 4)
    assert z_sag == pytest.approx((1.0, 2.0, 3.0)) and z_ax == pytest.approx((3.0, 2.0, 1.0))
    assert ax[5, 2, 1] == 3.0
    with pytest.raises(ValueError):
        load_canonical_nifti(path, plane="coronal")
