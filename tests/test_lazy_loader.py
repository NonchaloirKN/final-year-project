"""
Unit tests for Task 1.2: Memory-Mapped Lazy Loading of 3D Volumetric Slices.

Verifies memory consumption bounds (< 50MB RSS increase), accurate metadata
parsing, and slice extraction across orthogonal axes for both SimpleITK and
Nibabel backends.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import tempfile
import psutil
import pytest
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from src.utils.data_loader import LazyVolumeReader, read_volume_slice


# Default real dataset path from the 3.3GB RESECT glioma cohort
REAL_VOLUME_PATH = Path("data/resect_base/RESECT/NIFTI/Case1/MRI/Case1-FLAIR.nii.gz")


@pytest.fixture(scope="session")
def sample_volume_path() -> Path:
    """
    Returns a path to a valid NIfTI 3D volume.
    Uses the real RESECT dataset if present; otherwise, synthesizes a temporary volume.
    """
    if REAL_VOLUME_PATH.exists():
        return REAL_VOLUME_PATH

    # Synthetic fallback for isolated environments
    temp_dir = tempfile.mkdtemp()
    synthetic_path = Path(temp_dir) / "synthetic_mri_flair.nii.gz"
    data = np.random.randn(64, 64, 64).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(synthetic_path))
    return synthetic_path


def test_lazy_memory_consumption(sample_volume_path: Path):
    """
    Assert that initializing the reader and reading volume header metadata
    consumes less than 50MB of RAM.
    """
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    # Initialize reader without reading full pixel data into RAM
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(sample_volume_path))
    reader.ReadImageInformation()
    size = reader.GetSize()

    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    mem_diff = mem_after - mem_before

    assert (
        mem_diff < 50.0
    ), f"Lazy loading exceeded memory limits! Consumed {mem_diff:.2f} MB"
    assert len(size) == 3, f"Failed to parse 3D volume information! Got size: {size}"


def test_lazy_volume_reader_sitk_backend(sample_volume_path: Path):
    """
    Test LazyVolumeReader using SimpleITK backend:
    - Verifies metadata parsing
    - Extracts 2D slices along X, Y, and Z axes
    - Asserts extracted slices are 2D NumPy arrays
    - Verifies memory consumption is < 50MB
    """
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    reader = LazyVolumeReader(backend="simpleitk")
    info = reader.get_volume_info(sample_volume_path)

    assert "size" in info
    assert len(info["size"]) == 3
    size_x, size_y, size_z = info["size"]

    # Extract axial slice (axis=2 / 'z')
    axial_idx = size_z // 2
    slice_z = reader.get_slice(sample_volume_path, slice_idx=axial_idx, axis=2)
    assert isinstance(slice_z, np.ndarray), "Slice must be a NumPy array."
    assert slice_z.ndim == 2, f"Expected 2D slice, got {slice_z.ndim}D array."
    assert slice_z.shape == (size_y, size_x), (
        f"Expected shape ({size_y}, {size_x}), got {slice_z.shape}"
    )

    # Extract coronal slice (axis=1 / 'coronal')
    coronal_idx = size_y // 2
    slice_y = reader.get_slice(
        sample_volume_path, slice_idx=coronal_idx, axis="coronal"
    )
    assert slice_y.ndim == 2
    assert slice_y.shape == (size_z, size_x)

    # Extract sagittal slice (axis=0 / 'sagittal')
    sagittal_idx = size_x // 2
    slice_x = reader.get_slice(
        sample_volume_path, slice_idx=sagittal_idx, axis="sagittal"
    )
    assert slice_x.ndim == 2
    assert slice_x.shape == (size_z, size_y)

    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_diff = mem_after - mem_before
    assert (
        mem_diff < 50.0
    ), f"SimpleITK slice extraction consumed too much memory: {mem_diff:.2f} MB"


def test_lazy_volume_reader_nibabel_backend(sample_volume_path: Path):
    """
    Test LazyVolumeReader using Nibabel backend with memory mapping:
    - Extracts 2D slice
    - Asserts 2D dimensionality
    - Verifies memory consumption is < 50MB
    """
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    reader = LazyVolumeReader(backend="nibabel")
    info = reader.get_volume_info(sample_volume_path)
    size_x, size_y, size_z = info["size"]

    axial_idx = size_z // 2
    slice_z = reader.get_slice(sample_volume_path, slice_idx=axial_idx, axis=2)

    assert isinstance(slice_z, np.ndarray), "Slice must be a NumPy array."
    assert slice_z.ndim == 2, f"Expected 2D slice, got {slice_z.ndim}D array."
    assert slice_z.shape == (size_x, size_y)

    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_diff = mem_after - mem_before
    assert (
        mem_diff < 50.0
    ), f"Nibabel slice extraction consumed too much memory: {mem_diff:.2f} MB"


def test_lazy_volume_reader_instance_binding(sample_volume_path: Path):
    """
    Test LazyVolumeReader initialized with a default volume path.
    """
    reader = LazyVolumeReader(volume_path=sample_volume_path, backend="simpleitk")
    size = reader.get_size()
    assert len(size) == 3

    # Call get_slice without passing volume_path
    slice_data = reader.get_slice(slice_idx=size[2] // 2, axis=2)
    assert isinstance(slice_data, np.ndarray)
    assert slice_data.ndim == 2


def test_convenience_function(sample_volume_path: Path):
    """
    Test read_volume_slice module-level convenience function.
    """
    slice_data = read_volume_slice(sample_volume_path, slice_idx=10, axis=2)
    assert isinstance(slice_data, np.ndarray)
    assert slice_data.ndim == 2


def test_boundary_and_error_handling(sample_volume_path: Path):
    """
    Test that invalid slice indices and non-existent files raise appropriate errors.
    """
    reader = LazyVolumeReader(volume_path=sample_volume_path)

    # Test negative index
    with pytest.raises(IndexError):
        reader.get_slice(slice_idx=-1, axis=2)

    # Test index out of range
    size = reader.get_size()
    with pytest.raises(IndexError):
        reader.get_slice(slice_idx=size[2] + 100, axis=2)

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        reader.get_slice(
            volume_path="non_existent_scan_12345.nii.gz", slice_idx=0, axis=2
        )

    # Test invalid axis
    with pytest.raises(ValueError):
        reader.get_slice(slice_idx=0, axis="invalid_axis")
