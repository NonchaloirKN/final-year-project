"""
Physical Evaluation Metrics for 6D Pose Estimation (Sprint 5, Task 5.2).

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Provides physical, clinical metric implementations for cross-subject pose evaluation:
1. compute_mte: Mean Translation Error (MTE) in physical millimeters (Euclidean distance).
2. compute_mae: Mean Angular Error (MAE) in degrees using the geodesic rotation distance.
3. evaluate_pose_predictions: High-level evaluation utility computing MTE, MAE, and component metrics.
4. volume_half_extent_mm: Physical scale of the normalised grid translations used by VolumeReslicer.
5. pose_unit_mm: Millimetres per pose-translation unit for the legacy or physical reslicer geometry.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.models.pose_regressor import rotation_6d_to_matrix


def _to_tensor(x: Union[torch.Tensor, np.ndarray], device: Optional[torch.device] = None) -> torch.Tensor:
    """Helper to convert input to float32 PyTorch tensor."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.tensor(x, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def volume_half_extent_mm(
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, float, float]:
    """
    Physical half-extent (mm) of the normalised grid along (x, y, z).

    physical_to_normalized_pose maps t_mm -> 2 * t_mm / ((N - 1) * s), so a normalised
    translation of 1.0 spans (N - 1) * s / 2 mm on that axis (127.5, 127.5, 95.5 mm for a
    192x256x256 volume at 1 mm). Pass the result as trans_bound_mm to compute_mte and
    evaluate_pose_predictions when translations come from sample_random_6d_poses or
    VolumeReslicer; the +/-20 mm sampling bound is not the normalisation scale.

    Args:
        volume_shape: (D, H, W) of the volume.
        spacing: (sz, sy, sx) physical voxel spacing in mm.

    Returns:
        (half_x_mm, half_y_mm, half_z_mm).
    """
    d, h, w = volume_shape
    sz, sy, sx = spacing
    return (
        float((w - 1) * sx / 2.0),
        float((h - 1) * sy / 2.0),
        float((d - 1) * sz / 2.0),
    )


def pose_unit_mm(
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ref_mm: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Millimetres per unit of the pose translation, per axis (x, y, z).

    Physical reslicer geometry (VolumeReslicer(ref_mm=...)): translations are stored in units of ref_mm on
    every axis, so the scale is (ref_mm, ref_mm, ref_mm) for every volume. Legacy geometry (ref_mm=None):
    translations are normalised grid coordinates, so the scale is the volume half-extent.
    """
    if ref_mm is not None:
        return (float(ref_mm), float(ref_mm), float(ref_mm))
    return volume_half_extent_mm(volume_shape, spacing)


def compute_mte(
    t_pred: Union[torch.Tensor, np.ndarray],
    t_gt: Union[torch.Tensor, np.ndarray],
    trans_bound_mm: Union[float, Tuple[float, float, float], torch.Tensor] = 20.0,
    is_normalized: bool = True,
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """
    Compute Mean Translation Error (MTE) in physical millimeters.

    Calculates the physical Euclidean distance between predicted and ground-truth
    translation vectors:
        MTE = || t_pred_mm - t_gt_mm ||_2

    Args:
        t_pred: Predicted translation or full pose tensor of shape (..., 3) or (..., 6) or (..., 9).
                If shape is (..., >3), the first 3 coordinates are extracted.
        t_gt: Ground-truth translation or full pose tensor of matching shape.
        trans_bound_mm: Physical mm spanned by a normalised translation of 1.0 (default: 20.0 mm).
                        If a scalar is provided, the same scale is used for all 3 axes; a 3-tuple gives
                        per-axis scales. For VolumeReslicer poses use volume_half_extent_mm(...).
        is_normalized: If True (default), inputs are assumed to be normalised in [-1, 1]
                       and are scaled by trans_bound_mm back to physical mm.
                       If False, inputs are already in physical mm.
        reduction: 'mean' (default) returns scalar mean error across the batch,
                   'none' returns per-sample error tensor of shape (...),
                   'median' returns median error.

    Returns:
        MTE in physical millimeters (scalar tensor if reduction='mean' and input is Tensor,
        or float if input was NumPy array).
    """
    input_is_numpy = isinstance(t_pred, np.ndarray) and isinstance(t_gt, np.ndarray)

    t_p = _to_tensor(t_pred)
    t_g = _to_tensor(t_gt, device=t_p.device)

    # Extract translation components (first 3 elements) if full pose was provided
    if t_p.shape[-1] > 3:
        t_p = t_p[..., :3]
    if t_g.shape[-1] > 3:
        t_g = t_g[..., :3]

    if t_p.shape != t_g.shape:
        raise ValueError(f"Shape mismatch in compute_mte: t_pred {t_p.shape} vs t_gt {t_g.shape}")

    # Un-normalise [-1, 1] to physical mm if requested
    if is_normalized:
        if isinstance(trans_bound_mm, (tuple, list)):
            scale = torch.tensor(trans_bound_mm, dtype=torch.float32, device=t_p.device)
        elif isinstance(trans_bound_mm, torch.Tensor):
            scale = trans_bound_mm.to(device=t_p.device, dtype=torch.float32)
        else:
            scale = float(trans_bound_mm)
        t_p_mm = t_p * scale
        t_g_mm = t_g * scale
    else:
        t_p_mm = t_p
        t_g_mm = t_g

    # Euclidean distance along translation coordinate axis
    # e_i = sqrt((x_p - x_g)^2 + (y_p - y_g)^2 + (z_p - z_g)^2)
    diff = t_p_mm - t_g_mm
    err = torch.linalg.norm(diff, ord=2, dim=-1)

    if reduction == "mean":
        res = err.mean()
    elif reduction == "median":
        res = err.median()
    elif reduction == "none":
        res = err
    else:
        raise ValueError(f"Unsupported reduction: {reduction}. Choose 'mean', 'median', or 'none'.")

    if input_is_numpy and reduction != "none":
        return float(res.item())
    elif input_is_numpy and reduction == "none":
        return res.cpu().numpy()
    return res


def compute_mae(
    R_pred: Union[torch.Tensor, np.ndarray],
    R_gt: Union[torch.Tensor, np.ndarray],
    eps: float = 0.0,
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """
    Compute Mean Angular Error (MAE) in degrees using the geodesic rotation distance on SO(3).

    Geodesic rotation formula:
        theta = arccos( clamp( (Tr(R_pred^T @ R_gt) - 1) / 2, -1 + eps, 1 - eps ) ) * (180 / pi)

    Args:
        R_pred: Predicted rotation. Can be:
                - 3x3 rotation matrices of shape (..., 3, 3)
                - 6D continuous rotation vectors of shape (..., 6)
                - 9D pose vectors of shape (..., 9) (extracts elements 3:9)
        R_gt: Ground-truth rotation matching representation or 3x3 matrices.
        eps: Numerical stability constant for arccos clamp (default: 0.0).
             When eps=0.0, clamps to [-1.0, 1.0] ensuring exact 0.0 error for identical rotations.
        reduction: 'mean' (default) returns scalar mean error in degrees,
                   'none' returns per-sample error tensor in degrees,
                   'median' returns median error in degrees.

    Returns:
        MAE in degrees (scalar tensor if reduction='mean' and input is Tensor,
        or float if input was NumPy array).
    """
    input_is_numpy = isinstance(R_pred, np.ndarray) and isinstance(R_gt, np.ndarray)

    r_p = _to_tensor(R_pred)
    r_g = _to_tensor(R_gt, device=r_p.device)

    # 1. Convert R_pred to (..., 3, 3) if needed
    if r_p.shape[-1] == 9:
        # Full 9D pose: elements 3:9 are continuous 6D rotation
        mat_p = rotation_6d_to_matrix(r_p[..., 3:9])
    elif r_p.shape[-1] == 6 and r_p.dim() < 3:
        # Continuous 6D rotation
        mat_p = rotation_6d_to_matrix(r_p)
    elif r_p.shape[-2:] == (3, 3):
        mat_p = r_p
    else:
        raise ValueError(f"Unrecognized shape for R_pred: {r_p.shape}. Expected (..., 3, 3), (..., 6), or (..., 9).")

    # 2. Convert R_gt to (..., 3, 3) if needed
    if r_g.shape[-1] == 9:
        mat_g = rotation_6d_to_matrix(r_g[..., 3:9])
    elif r_g.shape[-1] == 6 and r_g.dim() < 3:
        mat_g = rotation_6d_to_matrix(r_g)
    elif r_g.shape[-2:] == (3, 3):
        mat_g = r_g
    else:
        raise ValueError(f"Unrecognized shape for R_gt: {r_g.shape}. Expected (..., 3, 3), (..., 6), or (..., 9).")

    # Ensure shape broadcast compatibility
    if mat_p.shape != mat_g.shape:
        raise ValueError(f"Shape mismatch in compute_mae: mat_p {mat_p.shape} vs mat_g {mat_g.shape}")

    # 3. Compute relative rotation matrix: R_rel = R_pred^T @ R_gt
    # Notice that Tr(R_pred^T @ R_gt) == (mat_p * mat_g).sum(dim=(-2, -1))
    # Using explicit matrix multiplication for direct alignment with the supervisor's brief:
    r_rel = torch.matmul(mat_p.transpose(-1, -2), mat_g)
    trace = torch.diagonal(r_rel, dim1=-2, dim2=-1).sum(dim=-1)

    # 4. Geodesic angle in radians with protective clamp
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta_rad = torch.acos(cos_theta)
    theta_deg = theta_rad * (180.0 / math.pi)

    if reduction == "mean":
        res = theta_deg.mean()
    elif reduction == "median":
        res = theta_deg.median()
    elif reduction == "none":
        res = theta_deg
    else:
        raise ValueError(f"Unsupported reduction: {reduction}. Choose 'mean', 'median', or 'none'.")

    if input_is_numpy and reduction != "none":
        return float(res.item())
    elif input_is_numpy and reduction == "none":
        return res.cpu().numpy()
    return res


def evaluate_pose_predictions(
    p_pred: Union[torch.Tensor, np.ndarray],
    p_gt: Union[torch.Tensor, np.ndarray],
    trans_bound_mm: Union[float, Tuple[float, float, float]] = 20.0,
    is_normalized: bool = True,
) -> Dict[str, float]:
    """
    High-level clinical evaluation utility computing MTE, MAE, and directional metrics.

    Args:
        p_pred: Predicted 9D poses or (t, R) tuple.
        p_gt: Ground-truth 9D poses or (t, R) tuple.
        trans_bound_mm: Physical mm per normalised translation unit, scalar or per-axis 3-tuple
                        (default: 20.0). For VolumeReslicer poses use volume_half_extent_mm(...).
        is_normalized: Whether translations are in [-1, 1].

    Returns:
        Dictionary with keys:
            - 'mte_mm': Mean Translation Error in mm.
            - 'mae_deg': Mean Angular Error in degrees.
            - 'tx_err_mm', 'ty_err_mm', 'tz_err_mm': Per-axis translation errors.
    """
    p_p = _to_tensor(p_pred)
    p_g = _to_tensor(p_gt, device=p_p.device)

    # Global MTE and MAE
    mte_val = float(compute_mte(p_p, p_g, trans_bound_mm=trans_bound_mm, is_normalized=is_normalized, reduction="mean").item())
    mae_val = float(compute_mae(p_p, p_g, reduction="mean").item())

    # Component-wise translation errors
    if is_normalized:
        scale = torch.as_tensor(trans_bound_mm, dtype=torch.float32, device=p_p.device)
    else:
        scale = torch.tensor(1.0, device=p_p.device)
    diff_t = torch.abs(p_p[..., :3] - p_g[..., :3]) * scale
    axis_errs = diff_t.mean(dim=0).cpu().tolist()

    return {
        "mte_mm": mte_val,
        "mae_deg": mae_val,
        "tx_err_mm": float(axis_errs[0]) if len(axis_errs) > 0 else 0.0,
        "ty_err_mm": float(axis_errs[1]) if len(axis_errs) > 1 else 0.0,
        "tz_err_mm": float(axis_errs[2]) if len(axis_errs) > 2 else 0.0,
    }
