"""
RESECTDataset & Preprocessing Utilities for FCN05 Medical Registration.

Implements:
- Task 1.3: Custom PyTorch RESECTDataset with modality-specific intensity normalization
  (US min-max scaling, MRI 99th-percentile clipping) and 227x227 tensor resizing.
- Task 1.4: Dynamic Bounding Box Jitter (apply_bbox_jitter) to strip artificial border shortcuts.
- Task 1.5: Blank-Slice Gating System (BlankSliceGate) to discard low-signal ultrasound frames.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils.data_loader import LazyVolumeReader


# ---------------------------------------------------------------------------
# Task 1.3 Helper Normalization Functions
# ---------------------------------------------------------------------------

def normalize_us(image: Union[np.ndarray, torch.Tensor], eps: float = 1e-8) -> np.ndarray:
    """
    Normalize 2D ultrasound slice intensities to [0, 1] via min-max scaling:
        I_norm = (I - I_min) / (I_max - I_min)

    Parameters
    ----------
    image : Union[np.ndarray, torch.Tensor]
        Raw ultrasound 2D image.
    eps : float, optional
        Numerical stability constant, by default 1e-8.

    Returns
    -------
    np.ndarray
        Normalized 2D array in range [0.0, 1.0].
    """
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image, dtype=np.float32)
    i_min = float(np.min(arr))
    i_max = float(np.max(arr))

    if np.isclose(i_min, i_max):
        return np.zeros_like(arr, dtype=np.float32)

    norm = (arr - i_min) / (i_max - i_min + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


CANONICAL_AXCODES: Tuple[str, str, str] = ("R", "A", "S")


SLICE_PLANES = ("sagittal", "axial")


def load_canonical_nifti(
    path: Union[str, Path],
    plane: str = "sagittal",
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[str, str, str]]:
    """
    Load a NIfTI volume reoriented to the closest canonical RAS+ orientation.

    RESECT stores 21 of its 23 T2-FLAIR volumes with array axes (R, A, S) but Case 4 as (P, I, R) and Case 11 as
    (A, S, R). nibabel.as_closest_canonical permutes and flips the array (no resampling) so that every volume has
    axis 0 = left-to-right, axis 1 = posterior-to-anterior and axis 2 = inferior-to-superior, and the voxel
    spacing is permuted with it. The same array index then means the same anatomical direction for every patient.

    The reslicer treats array axis 0 as the elevational (D) axis, so the canonical slice is the plane of axes 1 and 2:
      plane="sagittal": RAS order (R, A, S); the canonical slice is mid-sagittal and elevation runs left-right.
      plane="axial":    axes permuted to (S, A, R); the canonical slice is mid-axial and elevation runs
                        inferior-superior. Grid (x, y, z) = (axis 2, axis 1, axis 0) = (R, A, S), a right-handed frame.

    Returns:
        data: float32 array in RAS+ axis order.
        zooms: voxel spacing (mm) per array axis after reorientation.
        original_axcodes: orientation codes of the file before reorientation.
    """
    import nibabel as nib

    img = nib.load(str(path))
    original = tuple(nib.aff2axcodes(img.affine))
    canon = nib.as_closest_canonical(img)
    data = canon.get_fdata(dtype=np.float32)
    zooms = tuple(float(z) for z in canon.header.get_zooms()[:3])
    if plane not in SLICE_PLANES:
        raise ValueError(f"plane must be one of {SLICE_PLANES}, got {plane}")
    if plane == "axial":
        data = np.ascontiguousarray(data.transpose(2, 1, 0))
        zooms = (zooms[2], zooms[1], zooms[0])
    return data, zooms, original


def normalize_mri(
    image: Union[np.ndarray, torch.Tensor],
    percentile: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize 2D MRI slice intensities by clipping the top 1% (to remove
    hyperintense fat/fluid artifacts) and scaling to [0, 1].

    Parameters
    ----------
    image : Union[np.ndarray, torch.Tensor]
        Raw MRI 2D slice.
    percentile : float, optional
        Upper intensity clipping percentile, by default 99.0.
    eps : float, optional
        Numerical stability constant, by default 1e-8.

    Returns
    -------
    np.ndarray
        Normalized 2D array in range [0.0, 1.0].
    """
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image, dtype=np.float32)
    i_min = float(np.min(arr))
    p_upper = float(np.percentile(arr, percentile))

    if np.isclose(i_min, p_upper) or p_upper <= i_min:
        i_max = float(np.max(arr))
        if np.isclose(i_min, i_max):
            return np.zeros_like(arr, dtype=np.float32)
        norm = (arr - i_min) / (i_max - i_min + eps)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    clipped = np.clip(arr, i_min, p_upper)
    norm = (clipped - i_min) / (p_upper - i_min + eps)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Task 1.4 Dynamic Bounding Box Jitter
# ---------------------------------------------------------------------------

def apply_bbox_jitter(
    base_roi: Sequence[int],
    min_jitter: int = 5,
    max_jitter: int = 10,
) -> Tuple[List[int], List[int]]:
    """
    Apply a random 3D spatial translation offset (jitter) to an ROI bounding box.

    Preserves bounding box volume while randomly shifting coordinate boundaries
    by Delta x, Delta y, Delta z where min_jitter <= |Delta| <= max_jitter.
    Prevents the neural network from relying on static cropping borders as an
    anatomical location shortcut.

    Parameters
    ----------
    base_roi : Sequence[int]
        Original bounding box coordinates: [x_min, y_min, z_min, x_max, y_max, z_max].
    min_jitter : int, optional
        Minimum jitter offset magnitude in voxels, by default 5.
    max_jitter : int, optional
        Maximum jitter offset magnitude in voxels, by default 10.

    Returns
    -------
    Tuple[List[int], List[int]]
        jittered_roi : [x_min+dx, y_min+dy, z_min+dz, x_max+dx, y_max+dy, z_max+dz]
        offset : [dx, dy, dz]
    """
    if len(base_roi) != 6:
        raise ValueError(f"base_roi must contain 6 coordinates, got {len(base_roi)}")

    x_min, y_min, z_min, x_max, y_max, z_max = base_roi

    offsets: List[int] = []
    for _ in range(3):
        mag = int(np.random.randint(min_jitter, max_jitter + 1))
        sign = int(np.random.choice([-1, 1]))
        offsets.append(sign * mag)

    dx, dy, dz = offsets
    jittered_roi = [
        int(x_min + dx),
        int(y_min + dy),
        int(z_min + dz),
        int(x_max + dx),
        int(y_max + dy),
        int(z_max + dz),
    ]

    return jittered_roi, offsets


# ---------------------------------------------------------------------------
# Task 1.5 Blank-Slice Gating System
# ---------------------------------------------------------------------------

def _evaluate_blank_slice(
    slice_tensor: Union[torch.Tensor, np.ndarray],
    active_pixel_val: float = 0.1,
    min_pixels: int = 500,
) -> bool:
    """Evaluate whether a slice contains sufficient active anatomical pixels."""
    if isinstance(slice_tensor, torch.Tensor):
        active_count = int(torch.sum(slice_tensor > active_pixel_val).item())
    elif isinstance(slice_tensor, np.ndarray):
        active_count = int(np.sum(slice_tensor > active_pixel_val))
    else:
        raise TypeError(
            f"Expected torch.Tensor or np.ndarray, got {type(slice_tensor).__name__}"
        )
    return active_count >= min_pixels


class _BlankSliceGateCallable:
    """Callable gating operator configured with threshold parameters."""

    def __init__(self, active_pixel_val: float = 0.1, min_pixels: int = 500) -> None:
        self.active_pixel_val = active_pixel_val
        self.min_pixels = min_pixels

    def __call__(
        self,
        slice_tensor: Union[torch.Tensor, np.ndarray],
        active_pixel_val: Optional[float] = None,
        min_pixels: Optional[int] = None,
    ) -> bool:
        val_thresh = (
            active_pixel_val if active_pixel_val is not None else self.active_pixel_val
        )
        px_thresh = min_pixels if min_pixels is not None else self.min_pixels
        return _evaluate_blank_slice(
            slice_tensor, active_pixel_val=val_thresh, min_pixels=px_thresh
        )


def BlankSliceGate(
    arg: Optional[Union[torch.Tensor, np.ndarray, float]] = None,
    active_pixel_val: float = 0.1,
    min_pixels: int = 500,
    **kwargs: Any,
) -> Union[bool, _BlankSliceGateCallable]:
    """
    Binary pre-filtering gate to identify and filter out empty or low-signal
    ultrasound slices near scan borders.

    Can be used both as an instantiated operator:
        gate = BlankSliceGate(active_pixel_val=0.1, min_pixels=500)
        is_active = gate(image_tensor)

    And directly as a functional evaluator:
        is_active = BlankSliceGate(image_tensor, min_pixels=500)
    """
    if isinstance(arg, (torch.Tensor, np.ndarray)):
        # Direct functional evaluation: BlankSliceGate(image_tensor, ...)
        val_thresh = kwargs.get("active_pixel_val", active_pixel_val)
        px_thresh = kwargs.get("min_pixels", min_pixels)
        return _evaluate_blank_slice(
            arg, active_pixel_val=val_thresh, min_pixels=px_thresh
        )

    # Operator instantiation: gate = BlankSliceGate(active_pixel_val=..., min_pixels=...)
    val_thresh = float(arg) if isinstance(arg, (int, float)) else active_pixel_val
    px_thresh = int(kwargs.get("min_pixels", min_pixels))
    return _BlankSliceGateCallable(
        active_pixel_val=val_thresh, min_pixels=px_thresh
    )


# ---------------------------------------------------------------------------
# RESECT Cohort Case Identifiers
# Note: Case9, Case10, Case20, and Case22 are skipped in the original RESECT dataset,
# resulting in 23 total clinical cases spanning Case1 through Case27.
# ---------------------------------------------------------------------------
RESECT_PATIENT_CASES: List[str] = [
    "Case1", "Case2", "Case3", "Case4", "Case5", "Case6", "Case7", "Case8",
    "Case11", "Case12", "Case13", "Case14", "Case15", "Case16", "Case17", "Case18", "Case19",
    "Case21", "Case23", "Case24", "Case25", "Case26", "Case27",
]


# ---------------------------------------------------------------------------
# Task 1.3 RESECTDataset Class
# ---------------------------------------------------------------------------

class RESECTDataset(Dataset):
    """
    PyTorch Dataset for the clinical RESECT database.

    Pairs 2D intraoperative ultrasound (iUS) slices with corresponding
    preoperative T2-FLAIR MRI (pMRI) slices. Applies lazy memory-mapped loading,
    modality-specific intensity normalization, and bilinear spatial resizing
    to [1, 227, 227].

    Parameters
    ----------
    data_dir : Union[str, Path], optional
        Root directory of the RESECT NIFTI files, by default 'data/resect_base/RESECT/NIFTI'.
    csv_file : Optional[Union[str, Path]], optional
        Path to cached slice-pairing CSV. If it exists, metadata is loaded from it.
        If it does not exist, metadata is automatically built and saved to it.
    resize : Optional[Tuple[int, int]], optional
        Output spatial dimensions (Height, Width), by default (227, 227).
    us_phase : str, optional
        Intraoperative ultrasound scan phase: 'before' (default), 'during', or 'after'.
    cases : Optional[Sequence[str]], optional
        Subset of case identifiers to include (e.g. ['Case1', 'Case2']).
    use_gating : bool, optional
        If True, filters out blank ultrasound slices with < 500 active pixels.
    transform : Optional[Callable], optional
        Optional torchvision-style transform callable applied to (us, mri).
    backend : str, optional
        Underlying I/O backend for LazyVolumeReader ('simpleitk' or 'nibabel').
    sample_stride : int, optional
        Sampling stride through axial slices when generating pairs, by default 2.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/resect_base/RESECT/NIFTI",
        csv_file: Optional[Union[str, Path]] = None,
        resize: Optional[Tuple[int, int]] = (227, 227),
        us_phase: str = "before",
        cases: Optional[Sequence[str]] = None,
        use_gating: bool = False,
        transform: Optional[Callable] = None,
        backend: str = "simpleitk",
        sample_stride: int = 2,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.csv_file = Path(csv_file) if csv_file is not None else None
        self.resize = resize
        self.us_phase = us_phase.lower()
        self.cases_filter = set(cases) if cases is not None else None
        self.use_gating = use_gating
        self.transform = transform
        self.reader = LazyVolumeReader(backend=backend)
        self.sample_stride = max(1, sample_stride)

        self.metadata: List[Dict[str, Any]] = []
        self._load_or_build_metadata()

    def _load_or_build_metadata(self) -> None:
        """Load metadata from CSV if available, otherwise construct from data_dir."""
        if self.csv_file is not None and self.csv_file.exists():
            self._load_metadata_from_csv(self.csv_file)
            return

        self._build_metadata_from_directory()

        if self.csv_file is not None:
            self._save_metadata_to_csv(self.csv_file)

    def _load_metadata_from_csv(self, path: Path) -> None:
        """Read pre-computed paired slice metadata from CSV file."""
        self.metadata.clear()
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row["patient_id"]
                if self.cases_filter is not None and patient_id not in self.cases_filter:
                    continue
                self.metadata.append(
                    {
                        "patient_id": patient_id,
                        "mri_path": row["mri_path"],
                        "us_path": row["us_path"],
                        "mri_slice_idx": int(row["mri_slice_idx"]),
                        "us_slice_idx": int(row["us_slice_idx"]),
                        "z_coord": float(row.get("z_coord", 0.0)),
                    }
                )

    def _save_metadata_to_csv(self, path: Path) -> None:
        """Save discovered slice pairs to CSV for fast future initialization."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "patient_id",
                "mri_path",
                "us_path",
                "mri_slice_idx",
                "us_slice_idx",
                "z_coord",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.metadata:
                writer.writerow(item)

    def _build_metadata_from_directory(self) -> None:
        """Discover and pair corresponding MRI and US axial slices across cases."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"RESECT data directory not found at {self.data_dir}")

        case_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("Case")],
            key=lambda x: int(x.name.replace("Case", "")) if x.name.replace("Case", "").isdigit() else 999,
        )

        gate = BlankSliceGate(active_pixel_val=0.1, min_pixels=500)

        for case_dir in case_dirs:
            case_id = case_dir.name
            if self.cases_filter is not None and case_id not in self.cases_filter:
                continue

            mri_file = case_dir / "MRI" / f"{case_id}-FLAIR.nii.gz"
            us_file = case_dir / "US" / f"{case_id}-US-{self.us_phase}.nii.gz"

            if not mri_file.exists() or not us_file.exists():
                continue

            try:
                mri_size = self.reader.get_size(mri_file)  # (X, Y, Z)
                us_size = self.reader.get_size(us_file)
            except Exception:
                continue

            mri_num_slices = mri_size[2]
            us_num_slices = us_size[2]

            # Central 80% volume range to avoid blank boundary scanning
            start_ratio, end_ratio = 0.10, 0.90
            start_mri = int(mri_num_slices * start_ratio)
            end_mri = int(mri_num_slices * end_ratio)

            for mri_idx in range(start_mri, end_mri, self.sample_stride):
                rel_pos = (mri_idx - start_mri) / max(1, (end_mri - start_mri))
                us_idx = int(us_num_slices * (start_ratio + rel_pos * (end_ratio - start_ratio)))
                us_idx = min(us_idx, us_num_slices - 1)

                if self.use_gating:
                    try:
                        raw_us = self.reader.get_slice(us_file, slice_idx=us_idx, axis=2)
                        norm_us = normalize_us(raw_us)
                        if not gate(norm_us):
                            continue
                    except Exception:
                        continue

                self.metadata.append(
                    {
                        "patient_id": case_id,
                        "mri_path": str(mri_file),
                        "us_path": str(us_file),
                        "mri_slice_idx": mri_idx,
                        "us_slice_idx": us_idx,
                        "z_coord": float(mri_idx),
                    }
                )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(f"Dataset index {idx} out of range [0, {len(self.metadata) - 1}]")

        item = self.metadata[idx]
        mri_path = item["mri_path"]
        us_path = item["us_path"]
        mri_idx = item["mri_slice_idx"]
        us_idx = item["us_slice_idx"]
        z_coord = item["z_coord"]

        # Lazy slice extraction from disk
        raw_mri = self.reader.get_slice(mri_path, slice_idx=mri_idx, axis=2)
        raw_us = self.reader.get_slice(us_path, slice_idx=us_idx, axis=2)

        # Modality-specific intensity normalization
        mri_norm = normalize_mri(raw_mri)
        us_norm = normalize_us(raw_us)

        # Convert to float32 tensors with channel dimension: [1, H, W]
        mri_tensor = torch.from_numpy(mri_norm).float().unsqueeze(0)
        us_tensor = torch.from_numpy(us_norm).float().unsqueeze(0)

        # Spatial resizing to [1, 227, 227] via bilinear interpolation
        if self.resize is not None:
            # F.interpolate requires [Batch, Channel, H, W]
            mri_tensor = F.interpolate(
                mri_tensor.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            us_tensor = F.interpolate(
                us_tensor.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Ensure strict [0.0, 1.0] numerical bounds
        mri_tensor = torch.clamp(mri_tensor, 0.0, 1.0)
        us_tensor = torch.clamp(us_tensor, 0.0, 1.0)

        # Continuous 9D pose ground-truth vector: [tx=0, ty=0, tz=z_coord, r1=1, r2=0, r3=0, r4=0, r5=1, r6=0]
        pose = torch.tensor(
            [0.0, 0.0, z_coord, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
        )

        if self.transform is not None:
            us_tensor, mri_tensor = self.transform(us_tensor, mri_tensor)

        return us_tensor, mri_tensor, pose
