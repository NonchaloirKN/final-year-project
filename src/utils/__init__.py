"""
Utility modules for data loading, transformations, and preprocessing.
"""

from src.utils.data_loader import LazyVolumeReader, read_volume_slice
from src.utils.dataset import (
    BlankSliceGate,
    RESECTDataset,
    RESECT_PATIENT_CASES,
    apply_bbox_jitter,
    normalize_mri,
    normalize_us,
)
from src.utils.metrics import compute_mae, compute_mte, evaluate_pose_predictions, pose_unit_mm, volume_half_extent_mm
from src.utils.reslicer import (
    VolumeReslicer,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    normalized_to_physical_pose,
    physical_to_normalized_pose,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

__all__ = [
    "LazyVolumeReader",
    "read_volume_slice",
    "RESECTDataset",
    "RESECT_PATIENT_CASES",
    "apply_bbox_jitter",
    "BlankSliceGate",
    "normalize_us",
    "normalize_mri",
    "VolumeReslicer",
    "axis_angle_to_matrix",
    "rotation_6d_to_matrix",
    "quaternion_to_matrix",
    "matrix_to_rotation_6d",
    "physical_to_normalized_pose",
    "normalized_to_physical_pose",
    "compute_mte",
    "compute_mae",
    "evaluate_pose_predictions",
    "volume_half_extent_mm",
    "pose_unit_mm",
]
