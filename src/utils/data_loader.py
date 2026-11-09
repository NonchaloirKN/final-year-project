"""
LazyVolumeReader: Memory-Mapped Lazy Loading of 3D Medical Imaging Volumes.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Supports the 3.3GB RESECT glioma database and scalable architecture for larger datasets
by avoiding full-volume RAM allocation and using streaming/memory-mapped slice extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk


class LazyVolumeReader:
    """
    Memory-efficient 3D volume reader for NIfTI (.nii, .nii.gz) medical volumes.

    Enables lazy loading and memory-mapping of volumetric MRI/US scans using
    either SimpleITK or Nibabel backends. Extracts individual 2D slices along
    arbitrary orthogonal planes (sagittal, coronal, axial) without loading
    the full 3D volume into host RAM.

    Parameters
    ----------
    volume_path : Optional[Union[str, Path]], optional
        Default path to a NIfTI volume file. If provided, instance methods can be
        called without re-specifying the path.
    backend : str, optional
        Underlying I/O engine to use: 'simpleitk' (default) or 'nibabel'.
    """

    AXIS_MAP = {
        0: 0,
        "x": 0,
        "sagittal": 0,
        1: 1,
        "y": 1,
        "coronal": 1,
        2: 2,
        "z": 2,
        "axial": 2,
    }

    def __init__(
        self,
        volume_path: Optional[Union[str, Path]] = None,
        backend: str = "simpleitk",
    ) -> None:
        self.backend = backend.lower()
        if self.backend not in ("simpleitk", "nibabel"):
            raise ValueError(
                f"Unsupported backend '{backend}'. Must be 'simpleitk' or 'nibabel'."
            )
        self.default_volume_path: Optional[Path] = (
            Path(volume_path) if volume_path is not None else None
        )

    def _resolve_volume_path(
        self, volume_path: Optional[Union[str, Path]]
    ) -> Path:
        """Resolve and validate the volumetric file path."""
        target_path = volume_path or self.default_volume_path
        if target_path is None:
            raise ValueError(
                "No volume path provided and no default volume set on instance."
            )
        path_obj = Path(target_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Volume file not found at: {path_obj}")
        return path_obj

    @classmethod
    def _normalize_axis(cls, axis: Union[int, str]) -> int:
        """Map integer or descriptive string axis names to 0, 1, or 2."""
        if isinstance(axis, str):
            normalized = axis.lower().strip()
            if normalized in cls.AXIS_MAP:
                return cls.AXIS_MAP[normalized]
            raise ValueError(
                f"Invalid axis string '{axis}'. Supported: 'x'/'sagittal' (0), "
                f"'y'/'coronal' (1), 'z'/'axial' (2)."
            )
        if isinstance(axis, int):
            if axis in (0, 1, 2):
                return axis
            raise ValueError(
                f"Invalid axis index {axis}. Must be 0 (X), 1 (Y), or 2 (Z)."
            )
        raise TypeError(f"Axis must be int or str, got {type(axis).__name__}")

    def get_volume_info(
        self, volume_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Read volumetric metadata (dimensions, voxel spacing, origin, direction)
        without reading the voxel grid into RAM.

        Parameters
        ----------
        volume_path : Optional[Union[str, Path]]
            Target NIfTI volume path.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'size' (X, Y, Z), 'spacing', 'origin',
            and 'direction' or 'affine'.
        """
        resolved_path = self._resolve_volume_path(volume_path)
        str_path = str(resolved_path)

        if self.backend == "simpleitk":
            reader = sitk.ImageFileReader()
            reader.SetFileName(str_path)
            reader.ReadImageInformation()
            size = reader.GetSize()  # (X, Y, Z)
            spacing = reader.GetSpacing()
            origin = reader.GetOrigin()
            direction = reader.GetDirection()
            pixel_type = sitk.GetPixelIDValueAsString(reader.GetPixelID())

            return {
                "size": size,
                "shape": (size[2], size[1], size[0]),  # (Z, Y, X) numpy convention
                "spacing": spacing,
                "origin": origin,
                "direction": direction,
                "pixel_type": pixel_type,
                "backend": "simpleitk",
                "file_path": str_path,
            }
        else:
            # nibabel backend
            img = nib.load(str_path, mmap=True)
            shape = img.shape  # (X, Y, Z) in NIfTI
            header = img.header
            zooms = header.get_zooms()[:3]
            affine = img.affine

            return {
                "size": shape,
                "shape": (shape[2], shape[1], shape[0]),
                "spacing": tuple(float(z) for z in zooms),
                "affine": affine,
                "dtype": str(header.get_data_dtype()),
                "backend": "nibabel",
                "file_path": str_path,
            }

    def get_size(
        self, volume_path: Optional[Union[str, Path]] = None
    ) -> Tuple[int, int, int]:
        """
        Return the 3D volume dimensions (size_x, size_y, size_z) without loading data.
        """
        info = self.get_volume_info(volume_path)
        return info["size"]

    def get_slice(
        self,
        volume_path: Optional[Union[str, Path]] = None,
        slice_idx: Optional[int] = None,
        axis: Union[int, str] = 2,
    ) -> np.ndarray:
        """
        Extract a single 2D slice from disk using memory-mapped or sub-region reading.

        Parameters
        ----------
        volume_path : Optional[Union[str, Path]]
            Target volume file. Can be omitted if default path was set at instantiation.
            Also supports passing `slice_idx` as first positional argument when volume_path
            is bound to instance.
        slice_idx : Optional[int]
            0-indexed slice coordinate along the chosen axis.
        axis : Union[int, str], optional
            Orthogonal cutting axis: 0 (Sagittal/X), 1 (Coronal/Y), or 2 (Axial/Z).
            Default is 2 (Axial).

        Returns
        -------
        np.ndarray
            2D NumPy array representing the extracted slice.

        Raises
        ------
        IndexError
            If slice_idx is negative or exceeds volume bounds along the given axis.
        FileNotFoundError
            If the volume file does not exist.
        """
        # Handle polymorphic argument passing:
        # e.g., reader.get_slice(slice_idx=50, axis=2) when default_volume_path is set,
        # or reader.get_slice(volume_path, slice_idx, axis).
        if isinstance(volume_path, int) and slice_idx is None:
            slice_idx = volume_path
            target_path = self._resolve_volume_path(None)
        else:
            target_path = self._resolve_volume_path(volume_path)

        if slice_idx is None:
            raise ValueError("slice_idx must be specified.")

        norm_axis = self._normalize_axis(axis)
        str_path = str(target_path)

        if self.backend == "simpleitk":
            reader = sitk.ImageFileReader()
            reader.SetFileName(str_path)
            reader.ReadImageInformation()
            size = list(reader.GetSize())  # [X, Y, Z]

            max_slices = size[norm_axis]
            if slice_idx < 0 or slice_idx >= max_slices:
                raise IndexError(
                    f"slice_idx {slice_idx} out of range [0, {max_slices - 1}] "
                    f"for axis {norm_axis} in volume {target_path.name}."
                )

            extract_index = [0, 0, 0]
            extract_size = list(size)

            extract_index[norm_axis] = int(slice_idx)
            extract_size[norm_axis] = 0  # Collapses dimension to 2D image

            reader.SetExtractIndex(extract_index)
            reader.SetExtractSize(extract_size)

            slice_img = reader.Execute()
            slice_arr = sitk.GetArrayFromImage(slice_img)
            return np.squeeze(slice_arr)

        else:
            # nibabel backend with memory-mapping (mmap=True)
            img = nib.load(str_path, mmap=True)
            shape = img.shape

            max_slices = shape[norm_axis]
            if slice_idx < 0 or slice_idx >= max_slices:
                raise IndexError(
                    f"slice_idx {slice_idx} out of range [0, {max_slices - 1}] "
                    f"for axis {norm_axis} in volume {target_path.name}."
                )

            # Lazy slice extraction using ArrayProxy
            if norm_axis == 0:
                raw_slice = img.dataobj[slice_idx, :, :]
            elif norm_axis == 1:
                raw_slice = img.dataobj[:, slice_idx, :]
            else:
                raw_slice = img.dataobj[:, :, slice_idx]

            slice_arr = np.asarray(raw_slice)
            return np.squeeze(slice_arr)


# Module-level convenience functions matching procedural API specifications
def read_volume_slice(
    volume_path: Union[str, Path],
    slice_idx: int,
    axis: Union[int, str] = 2,
    backend: str = "simpleitk",
) -> np.ndarray:
    """
    Convenience function for lazy-loading a 2D slice from a 3D NIfTI volume.
    """
    reader = LazyVolumeReader(backend=backend)
    return reader.get_slice(volume_path=volume_path, slice_idx=slice_idx, axis=axis)
