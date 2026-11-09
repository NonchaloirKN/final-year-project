"""
Differentiable 3D Volume Reslicing Engine (SE(3) Grid Sampling).

This module implements a continuous, fully differentiable 3D planar reslicer
for extracting 2D cross-sectional slices from 3D MRI volumes at arbitrary 6D poses
(3D translation + 3D rotation). It supports continuous rotation parameterizations
(axis-angle via Rodrigues' formula, continuous 6D rotation representations,
and unit quaternions) to eliminate Euler angle gimbal lock singularities and guarantee
well-behaved autograd backpropagation for end-to-end slice-to-volume pose estimation.
"""

from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical physical half-width (mm) of the slice and unit of the pose translation in physical mode:
# 127.5 mm = half the field of view of a 256-voxel axis at 1 mm, so a pose translation of 1.0 is 127.5 mm on
# every axis and every patient, and the 227 x 227 slice always spans 255 mm x 255 mm.
PHYSICAL_REF_MM: float = 127.5


def axis_angle_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """
    Convert an axis-angle rotation vector r in R^3 to a 3x3 rotation matrix
    using Rodrigues' rotation formula with numerical Taylor expansions around zero
    to guarantee finite, non-NaN gradients everywhere.

    Args:
        r: Tensor of shape (..., 3) representing axis-angle vectors where
           direction indicates rotation axis and magnitude indicates angle in radians.

    Returns:
        R: Tensor of shape (..., 3, 3) representing valid SO(3) rotation matrices.
    """
    orig_shape = r.shape[:-1]
    r_flat = r.reshape(-1, 3)
    
    theta_sq = (r_flat ** 2).sum(dim=-1, keepdim=True)  # (N, 1)
    theta_clamped = torch.sqrt(torch.clamp(theta_sq, min=1e-8))
    
    # Taylor expansion for small theta (theta < 1e-3)
    # sin(theta)/theta = 1 - theta^2/6 + theta^4/120
    # (1 - cos(theta))/theta^2 = 1/2 - theta^2/24 + theta^4/720
    alpha_small = 1.0 - theta_sq / 6.0 + (theta_sq ** 2) / 120.0
    beta_small = 0.5 - theta_sq / 24.0 + (theta_sq ** 2) / 720.0
    
    alpha_large = torch.sin(theta_clamped) / theta_clamped
    beta_large = (1.0 - torch.cos(theta_clamped)) / (theta_clamped ** 2)
    
    use_taylor = (theta_sq < 1e-6)
    alpha = torch.where(use_taylor, alpha_small, alpha_large).unsqueeze(-1)  # (N, 1, 1)
    beta = torch.where(use_taylor, beta_small, beta_large).unsqueeze(-1)    # (N, 1, 1)
    
    rx = r_flat[..., 0]
    ry = r_flat[..., 1]
    rz = r_flat[..., 2]
    zeros = torch.zeros_like(rx)
    
    # Skew-symmetric cross product matrix K
    # [ 0, -rz,  ry]
    # [ rz,   0, -rx]
    # [-ry,  rx,   0]
    K = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=-1).reshape(-1, 3, 3)
    
    K2 = torch.bmm(K, K)
    I = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0).expand_as(K)
    
    R = I + alpha * K + beta * K2
    return R.reshape(*orig_shape, 3, 3)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous 6D rotation representation (Zhou et al., CVPR 2019)
    to 3x3 rotation matrices using Gram-Schmidt orthogonalization.

    Args:
        d6: Tensor of shape (..., 6) where each row contains two 3D column candidates [a1, a2].

    Returns:
        R: Tensor of shape (..., 3, 3) representing orthogonal SO(3) rotation matrices.
    """
    orig_shape = d6.shape[:-1]
    d6_flat = d6.reshape(-1, 6)
    
    a1 = d6_flat[..., :3]
    a2 = d6_flat[..., 3:6]
    
    b1 = a1 / torch.clamp(torch.norm(a1, dim=-1, keepdim=True), min=1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2_raw = a2 - dot * b1
    b2 = b2_raw / torch.clamp(torch.norm(b2_raw, dim=-1, keepdim=True), min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    
    R = torch.stack([b1, b2, b3], dim=-1)  # Columns b1, b2, b3
    return R.reshape(*orig_shape, 3, 3)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Extract continuous 6D rotation representation (first two columns) from 3x3 rotation matrix.

    Args:
        R: Tensor of shape (..., 3, 3)

    Returns:
        d6: Tensor of shape (..., 6)
    """
    # Columns 0 and 1 flattened: [R[..., 0, 0], R[..., 1, 0], R[..., 2, 0], R[..., 0, 1], R[..., 1, 1], R[..., 2, 1]]
    col1 = R[..., :, 0]
    col2 = R[..., :, 1]
    return torch.cat([col1, col2], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions [qw, qx, qy, qz] to 3x3 rotation matrices.

    Args:
        q: Tensor of shape (..., 4) where q = [qw, qx, qy, qz].

    Returns:
        R: Tensor of shape (..., 3, 3).
    """
    orig_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4)
    q_norm = q_flat / torch.clamp(torch.norm(q_flat, dim=-1, keepdim=True), min=1e-8)
    
    qw = q_norm[..., 0]
    qx = q_norm[..., 1]
    qy = q_norm[..., 2]
    qz = q_norm[..., 3]
    
    r00 = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx ** 2 + qz ** 2)
    r12 = 2.0 * (qy * qz - qx * qw)
    
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
    
    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).reshape(-1, 3, 3)
    
    return R.reshape(*orig_shape, 3, 3)


def physical_to_normalized_pose(
    pose_mm: torch.Tensor,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Convert translation vector in physical millimeters to normalized PyTorch coordinates [-1, 1].

    Args:
        pose_mm: Tensor of shape (..., 6) with translations [tx_mm, ty_mm, tz_mm, ...].
        volume_shape: (D, H, W) of the volume.
        spacing: (sz, sy, sx) physical voxel spacing in mm.

    Returns:
        pose_norm: Tensor of same shape with normalized translations.
    """
    d, h, w = volume_shape
    sz, sy, sx = spacing
    extent_x = (w - 1) * sx
    extent_y = (h - 1) * sy
    extent_z = (d - 1) * sz
    
    pose_norm = pose_mm.clone()
    # Normalize physical offsets to [-1, 1] relative to volume center
    pose_norm[..., 0] = 2.0 * pose_mm[..., 0] / max(extent_x, 1e-6)
    pose_norm[..., 1] = 2.0 * pose_mm[..., 1] / max(extent_y, 1e-6)
    pose_norm[..., 2] = 2.0 * pose_mm[..., 2] / max(extent_z, 1e-6)
    return pose_norm


def normalized_to_physical_pose(
    pose_norm: torch.Tensor,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Convert translation vector in normalized PyTorch coordinates [-1, 1] to physical millimeters.

    Args:
        pose_norm: Tensor of shape (..., 6) with normalized translations.
        volume_shape: (D, H, W) of the volume.
        spacing: (sz, sy, sx) physical voxel spacing in mm.

    Returns:
        pose_mm: Tensor of same shape with physical millimeter translations.
    """
    d, h, w = volume_shape
    sz, sy, sx = spacing
    extent_x = (w - 1) * sx
    extent_y = (h - 1) * sy
    extent_z = (d - 1) * sz

    pose_mm = pose_norm.clone()
    pose_mm[..., 0] = pose_norm[..., 0] * extent_x / 2.0
    pose_mm[..., 1] = pose_norm[..., 1] * extent_y / 2.0
    pose_mm[..., 2] = pose_norm[..., 2] * extent_z / 2.0
    return pose_mm


class VolumeReslicer(nn.Module):
    """
    Differentiable 3D Volume Reslicing Engine using PyTorch grid_sample.

    Extracts 2D planar cross-sections from a 3D volumetric tensor [B, 1, D, H, W]
    at arbitrary 6D poses [tx, ty, tz, rx, ry, rz] using trilinear interpolation.

    Coordinate Conventions:
        - Volume spatial dimensions: (D, H, W)
        - PyTorch grid coordinates: (x, y, z) in normalized [-1, 1] space,
          where x maps to W, y maps to H, and z maps to D.
        - Canonical 2D slice plane (at zero pose):
          horizontal axis u in [-1, 1] (x / Width),
          vertical axis v in [-1, 1] (y / Height),
          out-of-plane coordinate w = 0 (z / Depth, corresponding exactly to center axial slice).
        - Legacy transformation (ref_mm=None): p_vol = R * p_can + t, applied in the normalised array frame.
          In millimetres this is S R S^-1 with S = diag(half-extent), which shears tilts when the
          half-extents differ, and the slice field of view and translation unit vary between volumes.
        - Physical transformation (ref_mm set, half_extent_mm passed per volume):
              p_mm   = R * (ref_mm * p_can) + ref_mm * t      (rigid motion in millimetres)
              p_norm = p_mm / half_extent_mm                  (grid_sample coordinates)
          so R is an exact rotation in millimetres, t is in units of ref_mm on every axis, and the
          slice spans 2 * ref_mm in every volume.
    """

    def __init__(
        self,
        slice_size: Optional[Tuple[int, int]] = None,
        rotation_mode: str = "auto",
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
        ref_mm: Optional[float] = None,
    ):
        """
        Args:
            slice_size: Optional tuple (H_slice, W_slice). If None, defaults to the (H, W)
                of the input volume.
            rotation_mode: "auto", "axis_angle" (6D: 3 translation + 3 axis-angle),
                "rotation_6d" (9D: 3 translation + 6 continuous rotation), or
                "quaternion" (7D: 3 translation + 4 quaternion).
            mode: Interpolation mode ('bilinear' for trilinear sampling, 'nearest').
            padding_mode: 'zeros', 'border', or 'reflection'.
            align_corners: Geometrically aligns corner pixels with -1 and 1 (default True).
            ref_mm: Physical half-width of the canonical slice and translation unit in mm. None (default)
                keeps the legacy normalised-frame geometry; set (e.g. PHYSICAL_REF_MM) for exact
                millimetre rigid geometry, which requires half_extent_mm at call time.
        """
        super().__init__()
        self.ref_mm = ref_mm
        self.slice_size = slice_size
        self.rotation_mode = rotation_mode
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def _resolve_rotation_matrix(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separate translation and convert rotation parameterization to 3x3 matrix.

        Args:
            pose: Tensor of shape (B, N) where N in {6, 7, 9}.

        Returns:
            t: Tensor of shape (B, 3) containing translations [tx, ty, tz].
            R: Tensor of shape (B, 3, 3) containing SO(3) rotation matrices.
        """
        b_dim = pose.shape[0]
        n_dim = pose.shape[-1]
        t = pose[..., :3]
        
        mode = self.rotation_mode
        if mode == "auto":
            if n_dim == 6:
                mode = "axis_angle"
            elif n_dim == 7:
                mode = "quaternion"
            elif n_dim == 9:
                mode = "rotation_6d"
            else:
                raise ValueError(
                    f"Unable to auto-detect rotation mode for pose tensor with {n_dim} elements. "
                    "Expected 6 (axis_angle), 7 (quaternion), or 9 (rotation_6d)."
                )

        if mode == "axis_angle":
            if n_dim < 6:
                raise ValueError(f"axis_angle requires at least 6 pose values, got {n_dim}")
            rot_params = pose[..., 3:6]
            R = axis_angle_to_matrix(rot_params)
        elif mode == "rotation_6d":
            if n_dim < 9:
                raise ValueError(f"rotation_6d requires at least 9 pose values [3 trans + 6 rot], got {n_dim}")
            rot_params = pose[..., 3:9]
            R = rotation_6d_to_matrix(rot_params)
        elif mode == "quaternion":
            if n_dim < 7:
                raise ValueError(f"quaternion requires at least 7 pose values [3 trans + 4 quat], got {n_dim}")
            rot_params = pose[..., 3:7]
            R = quaternion_to_matrix(rot_params)
        else:
            raise ValueError(f"Unsupported rotation mode '{mode}'. Choose 'axis_angle', 'rotation_6d', or 'quaternion'.")

        return t, R

    def compute_grid(
        self,
        pose: torch.Tensor,
        volume_shape: Tuple[int, ...],
        slice_size: Optional[Tuple[int, int]] = None,
        half_extent_mm: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Generate the 3D sampling grid in normalized coordinates [-1, 1] for grid_sample.

        Args:
            pose: Tensor of shape (B, P) representing poses.
            volume_shape: Shape tuple of the 3D volume (..., D, H, W).
            slice_size: Optional (H_out, W_out). Defaults to (H, W) from volume_shape.

        Returns:
            grid: Tensor of shape (B, 1, H_out, W_out, 3) ready for torch.nn.functional.grid_sample.
        """
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
            
        b = pose.shape[0]
        device = pose.device
        dtype = pose.dtype
        
        d_vol, h_vol, w_vol = volume_shape[-3:]
        
        target_size = slice_size or self.slice_size or (h_vol, w_vol)
        h_slice, w_slice = target_size
        
        # 1. Canonical 2D planar slice coordinate mesh in normalized space [-1, 1]
        # x across columns (width), y across rows (height), z = 0 (axial plane)
        xs = torch.linspace(-1.0, 1.0, w_slice, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, h_slice, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_z = torch.zeros_like(grid_x)
        
        # Canonical points: shape (H_slice * W_slice, 3)
        can_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        
        # 2. Extract translation and rotation matrix
        t, R = self._resolve_rotation_matrix(pose)  # t: (B, 3), R: (B, 3, 3)
        
        # 3. Transform points: p_vol = R * p_can + t
        # In batch matrix multiplication:
        # can_pts: (1, HW, 3), R^T: (B, 3, 3) -> can_pts @ R^T: (B, HW, 3)
        pts_rotated = torch.matmul(can_pts.unsqueeze(0), R.transpose(1, 2))  # (B, HW, 3)
        pts_transformed = pts_rotated + t.unsqueeze(1)                        # (B, HW, 3)

        # 4. Physical mode: the motion above is in units of ref_mm, so p_mm = ref_mm * pts; map to the
        #    grid frame with the volume half-extent (x, y, z order), p_norm = p_mm / half_extent_mm
        if self.ref_mm is not None:
            if half_extent_mm is None:
                raise ValueError("VolumeReslicer(ref_mm=...) requires half_extent_mm=(hx, hy, hz) for each call")
            h = torch.as_tensor(half_extent_mm, device=device, dtype=dtype).reshape(-1, 1, 3)
            pts_transformed = pts_transformed * (self.ref_mm / h)
        
        # Reshape to (B, 1, H_slice, W_slice, 3) for 3D grid_sample
        grid = pts_transformed.reshape(b, 1, h_slice, w_slice, 3)
        return grid

    def forward(
        self,
        volume: torch.Tensor,
        pose: torch.Tensor,
        slice_size: Optional[Tuple[int, int]] = None,
        half_extent_mm: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extract 2D planar slices from 3D volume at specified continuous poses.

        Args:
            volume: 3D MRI volume tensor of shape [B, 1, D, H, W], [1, 1, D, H, W],
                    [1, D, H, W], or [D, H, W].
            pose: Continuous pose tensor of shape [B, 6] (or [6] for single slice),
                  [B, 7] (quaternion), or [B, 9] (rotation_6d).
            slice_size: Optional tuple (H_slice, W_slice). Defaults to volume (H, W).

        Returns:
            slice_2d: Extracted 2D slice tensor of shape [B, 1, H_slice, W_slice].
        """
        # Ensure volume is a torch.Tensor
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)

        # Ensure volume is 5D: [B_vol, C, D, H, W]
        if volume.dim() == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.dim() == 4:
            volume = volume.unsqueeze(1)
        elif volume.dim() != 5:
            raise ValueError(f"Expected 3D, 4D, or 5D volume tensor, got shape {volume.shape}")

        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        b_pose = pose.shape[0]
        b_vol = volume.shape[0]

        # Broadcast single volume across multiple poses if needed
        if b_vol == 1 and b_pose > 1:
            volume = volume.expand(b_pose, -1, -1, -1, -1)
        elif b_vol != b_pose:
            raise ValueError(
                f"Batch size mismatch: volume batch ({b_vol}) and pose batch ({b_pose}) "
                "must be identical, or volume batch must be 1."
            )

        # Cast pose to matching device and float dtype if needed
        if pose.device != volume.device or pose.dtype != volume.dtype:
            pose = pose.to(device=volume.device, dtype=volume.dtype)

        # Generate sampling grid
        grid = self.compute_grid(pose, volume.shape, slice_size=slice_size, half_extent_mm=half_extent_mm)

        # Trilinear sampling via 3D grid_sample
        # Input: [B, C, D_in, H_in, W_in]
        # Grid:  [B, 1, H_out, W_out, 3]
        # Output: [B, C, 1, H_out, W_out]
        sampled = F.grid_sample(
            volume,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        # Squeeze depth dimension out to return [B, C, H_out, W_out]
        return sampled.squeeze(2)

    def reslice(
        self,
        volume: torch.Tensor,
        pose: torch.Tensor,
        slice_size: Optional[Tuple[int, int]] = None,
        half_extent_mm: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Alias for forward pass (half_extent_mm is required in physical mode, as for forward)."""
        return self.forward(volume, pose, slice_size=slice_size, half_extent_mm=half_extent_mm)
