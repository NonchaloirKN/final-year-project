"""
Unit test suite for Differentiable 3D Volume Reslicing Engine (VolumeReslicer).

Validates:
1. Center axial slice extraction at zero pose (identity rotation, zero translation).
2. End-to-end autograd differentiability (backward pass computes finite, non-NaN gradients for all 6 pose parameters).
3. Translation shifts along depth (Z) axis matching geometric expectations.
4. In-plane rotation fidelity against discrete torch.rot90.
5. Mathematical validity of rotation representations (axis-angle, continuous 6D, quaternion).
6. Batch broadcasting (single volume resliced across a batch of candidate poses).
7. Custom output slice dimensions (e.g. 227x227 for ResNet feature extractors).
8. Physical millimeter to normalized coordinate conversions.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from src.utils.reslicer import (
    VolumeReslicer,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    normalized_to_physical_pose,
    physical_to_normalized_pose,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


def test_center_axial_slice_at_zero_pose():
    """
    Assert that zero-translation and zero-rotation extracts the exact center axial slice.
    """
    reslicer = VolumeReslicer()
    depth = 33
    h, w = 32, 32
    volume = torch.randn(1, 1, depth, h, w)
    zero_pose = torch.zeros(1, 6)  # [tx=0, ty=0, tz=0, rx=0, ry=0, rz=0]

    slice_out = reslicer(volume, zero_pose)
    center_idx = depth // 2  # 16 for D=33
    center_axial = volume[:, :, center_idx, :, :]

    # Single-precision grid_sample has slight float rounding (~1e-5), so check within 1e-4
    assert slice_out.shape == (1, 1, h, w), f"Expected shape (1, 1, {h}, {w}), got {slice_out.shape}"
    assert torch.allclose(slice_out, center_axial, atol=1e-4), (
        f"Zero pose slice differs from center axial slice! Max diff: {(slice_out - center_axial).abs().max().item()}"
    )

    # In double precision, verify high-fidelity agreement
    reslicer_d = VolumeReslicer(mode="bilinear")
    vol_double = volume.double()
    zero_pose_d = zero_pose.double()
    slice_out_d = reslicer_d(vol_double, zero_pose_d)
    center_axial_d = vol_double[:, :, center_idx, :, :]
    assert torch.allclose(slice_out_d, center_axial_d, atol=1e-7), "Double-precision center slice mismatch!"


def test_volume_reslicer_differentiability():
    """
    Assert that gradients flow backwards through the grid sampler into all 6 continuous pose parameters.
    """
    reslicer = VolumeReslicer()
    d, h, w = 33, 32, 32
    
    # Create volume with spatial intensity gradients so spatial derivatives are non-zero
    z_coords = torch.linspace(0, 1, d).reshape(1, 1, d, 1, 1).expand(1, 1, d, h, w)
    y_coords = torch.linspace(0, 1, h).reshape(1, 1, 1, h, 1).expand(1, 1, d, h, w)
    x_coords = torch.linspace(0, 1, w).reshape(1, 1, 1, 1, w).expand(1, 1, d, h, w)
    volume = z_coords + 2.0 * y_coords + 3.0 * x_coords

    # Perturbed 6D pose requiring gradients
    pose = torch.tensor([[0.05, -0.02, 0.08, 0.04, -0.03, 0.06]], requires_grad=True)

    slice_out = reslicer(volume, pose)
    target = volume[:, :, d // 2, :, :]
    loss = F.mse_loss(slice_out, target)
    loss.backward()

    assert pose.grad is not None, "Pose gradient was not computed!"
    assert not torch.isnan(pose.grad).any(), "NaN detected in pose gradients!"
    assert not torch.isinf(pose.grad).any(), "Inf detected in pose gradients!"
    assert (pose.grad != 0).all(), f"Expected non-zero gradients across all 6 parameters, got: {pose.grad}"


def test_volume_reslicer_translation_shifts():
    """
    Verify that translating along Z in normalized space shifts the sampled slice plane.
    """
    reslicer = VolumeReslicer()
    d, h, w = 33, 32, 32
    # Volume where slice intensity equals its z-index
    volume = torch.arange(d, dtype=torch.float32).reshape(1, 1, d, 1, 1).expand(1, 1, d, h, w).clone()

    # In normalized coordinates with align_corners=True, normalized step is 2 / (d - 1)
    dz_norm = 2.0 / (d - 1)
    
    # Shift forward by +2 slices
    pose_shift_pos = torch.tensor([[0.0, 0.0, 2.0 * dz_norm, 0.0, 0.0, 0.0]])
    slice_pos = reslicer(volume, pose_shift_pos)
    expected_pos = 16.0 + 2.0  # index 18
    assert torch.allclose(slice_pos, torch.full_like(slice_pos, expected_pos), atol=1e-3)

    # Shift backward by -3 slices
    pose_shift_neg = torch.tensor([[0.0, 0.0, -3.0 * dz_norm, 0.0, 0.0, 0.0]])
    slice_neg = reslicer(volume, pose_shift_neg)
    expected_neg = 16.0 - 3.0  # index 13
    assert torch.allclose(slice_neg, torch.full_like(slice_neg, expected_neg), atol=1e-3)


def test_volume_reslicer_rotations():
    """
    Verify in-plane rotation against discrete 90-degree tensor rotation.
    """
    reslicer = VolumeReslicer()
    d, h, w = 33, 32, 32
    volume = torch.randn(1, 1, d, h, w)
    
    # In-plane rotation around Z axis by +pi/2 radians (90 degrees counter-clockwise)
    pose_rot90 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0]])
    slice_rot = reslicer(volume, pose_rot90)

    center_slice = volume[0, 0, d // 2, :, :]
    # torch.rot90 rotates in plane (dims 0, 1) counter-clockwise
    expected_rot = torch.rot90(center_slice, k=1, dims=(0, 1))

    # Bilinear sampling vs discrete rotation: interior pixels should match closely
    # Exclude outer edge boundary pixels where sampling border effects occur
    interior_sampled = slice_rot[0, 0, 2:-2, 2:-2]
    interior_expected = expected_rot[2:-2, 2:-2]
    diff = (interior_sampled - interior_expected).abs().mean()
    assert diff.item() < 0.15, f"In-plane rotation mismatch! Mean difference: {diff.item()}"


def test_rotation_representations_validity():
    """
    Verify SO(3) matrix validity (orthogonality R * R^T = I, det(R) = +1)
    for axis-angle, continuous 6D, and quaternion representations.
    """
    torch.manual_seed(42)
    batch_size = 8

    # 1. Axis-Angle
    r = torch.randn(batch_size, 3)
    R_aa = axis_angle_to_matrix(r)
    # Check R * R^T = I
    I = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1)
    assert torch.allclose(torch.bmm(R_aa, R_aa.transpose(1, 2)), I, atol=1e-5), "Axis-angle not orthogonal!"
    # Check det(R) = +1
    dets_aa = torch.linalg.det(R_aa)
    assert torch.allclose(dets_aa, torch.ones(batch_size), atol=1e-5), "Axis-angle det != 1!"

    # Zero vector case
    R_zero = axis_angle_to_matrix(torch.zeros(1, 3))
    assert torch.allclose(R_zero, torch.eye(3).unsqueeze(0), atol=1e-6), "Zero axis-angle not identity!"

    # 2. Continuous 6D representation (Zhou et al. 2019)
    d6 = torch.randn(batch_size, 6)
    R_6d = rotation_6d_to_matrix(d6)
    assert torch.allclose(torch.bmm(R_6d, R_6d.transpose(1, 2)), I, atol=1e-5), "6D rotation not orthogonal!"
    dets_6d = torch.linalg.det(R_6d)
    assert torch.allclose(dets_6d, torch.ones(batch_size), atol=1e-5), "6D rotation det != 1!"

    # Identity 6D
    d6_id = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    assert torch.allclose(rotation_6d_to_matrix(d6_id), torch.eye(3).unsqueeze(0), atol=1e-6)

    # 3. Quaternions
    q = torch.randn(batch_size, 4)
    R_q = quaternion_to_matrix(q)
    assert torch.allclose(torch.bmm(R_q, R_q.transpose(1, 2)), I, atol=1e-5), "Quaternion not orthogonal!"
    dets_q = torch.linalg.det(R_q)
    assert torch.allclose(dets_q, torch.ones(batch_size), atol=1e-5), "Quaternion det != 1!"


def test_volume_reslicer_batch_broadcasting():
    """
    Verify that a single 3D volume [1, 1, D, H, W] is automatically broadcast across a batch of B poses.
    """
    reslicer = VolumeReslicer()
    volume = torch.randn(1, 1, 33, 32, 32)
    b = 5
    poses = torch.randn(b, 6) * 0.1  # 5 random poses

    slices = reslicer(volume, poses)
    assert slices.shape == (b, 1, 32, 32), f"Expected shape ({b}, 1, 32, 32), got {slices.shape}"


def test_volume_reslicer_custom_slice_size():
    """
    Verify reslicing with user-specified output slice dimensions (e.g. 227x227).
    """
    reslicer = VolumeReslicer(slice_size=(227, 227))
    volume = torch.randn(1, 1, 33, 64, 64)
    pose = torch.zeros(1, 6)

    slice_out = reslicer(volume, pose)
    assert slice_out.shape == (1, 1, 227, 227), f"Expected shape (1, 1, 227, 227), got {slice_out.shape}"


def test_physical_normalized_pose_conversion():
    """
    Verify round-trip conversion between physical millimeter coordinates and normalized coordinates.
    """
    volume_shape = (100, 200, 200)  # D, H, W
    spacing = (1.5, 1.0, 1.0)       # sz, sy, sx mm

    # Physical translation of 15mm in x, -20mm in y, 30mm in z
    pose_mm = torch.tensor([[15.0, -20.0, 30.0, 0.1, -0.2, 0.3]])
    pose_norm = physical_to_normalized_pose(pose_mm, volume_shape, spacing)

    # Check bounds
    assert -1.0 <= pose_norm[0, 0].item() <= 1.0
    assert -1.0 <= pose_norm[0, 1].item() <= 1.0
    assert -1.0 <= pose_norm[0, 2].item() <= 1.0

    # Invert back to physical
    pose_recon = normalized_to_physical_pose(pose_norm, volume_shape, spacing)
    assert torch.allclose(pose_mm, pose_recon, atol=1e-5), "Physical pose round-trip failed!"


def test_reslicer_rotation_modes():
    """
    Verify that VolumeReslicer operates with explicit rotation_mode arguments.
    """
    volume = torch.randn(1, 1, 33, 32, 32)

    # 1. 6D continuous rotation (9D pose: 3 trans + 6 rot)
    reslicer_6d = VolumeReslicer(rotation_mode="rotation_6d")
    pose_6d = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    slice_6d = reslicer_6d(volume, pose_6d)
    center_axial = volume[:, :, 16, :, :]
    assert torch.allclose(slice_6d, center_axial, atol=1e-4)

    # 2. Quaternion rotation (7D pose: 3 trans + 4 quat)
    reslicer_quat = VolumeReslicer(rotation_mode="quaternion")
    pose_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    slice_quat = reslicer_quat(volume, pose_quat)
    assert torch.allclose(slice_quat, center_axial, atol=1e-4)


def test_input_shape_flexibility():
    """
    Verify reslicer handles 3D, 4D, and 5D volumes as well as 1D pose vectors.
    """
    reslicer = VolumeReslicer()
    
    # 3D volume [D, H, W]
    vol_3d = torch.randn(33, 32, 32)
    pose_1d = torch.zeros(6)
    out_3d = reslicer(vol_3d, pose_1d)
    assert out_3d.shape == (1, 1, 32, 32)

    # 4D volume [1, D, H, W]
    vol_4d = torch.randn(1, 33, 32, 32)
    out_4d = reslicer(vol_4d, pose_1d)
    assert out_4d.shape == (1, 1, 32, 32)


def test_physical_mm_geometry_is_rigid_for_anisotropic_volumes():
    """
    In physical mode the sampled points, mapped back to millimetres, must equal R * (ref * p_can) + ref * t
    exactly, so distances and angles are preserved even when the half-extents differ (no S R S^-1 shear).
    """
    from src.utils.reslicer import PHYSICAL_REF_MM
    import math as _m
    reslicer = VolumeReslicer(slice_size=(9, 9), rotation_mode="axis_angle", ref_mm=PHYSICAL_REF_MM)
    shape = (1, 1, 32, 64, 64)
    half = (63 * 2.0 / 2.0, 63 * 2.0 / 2.0, 31 * 3.0 / 2.0)   # anisotropic: 63, 63, 46.5 mm
    angle = _m.radians(25.0)
    pose = torch.tensor([[0.05, -0.02, 0.1, angle, 0.0, 0.0]])   # tilt about x
    grid = reslicer.compute_grid(pose, shape, half_extent_mm=half)[0, 0].reshape(-1, 3)
    p_mm = grid * torch.tensor(half)

    us = torch.linspace(-1, 1, 9)
    gy, gx = torch.meshgrid(us, us, indexing="ij")
    p_can_mm = torch.stack([gx, gy, torch.zeros_like(gx)], dim=-1).reshape(-1, 3) * PHYSICAL_REF_MM
    R = axis_angle_to_matrix(pose[:, 3:6])[0]
    expected = p_can_mm @ R.T + pose[0, :3] * PHYSICAL_REF_MM
    assert torch.allclose(p_mm, expected, atol=1e-3)

    # Rigid: distance between the two in-plane neighbours along v is unchanged by the tilt
    step = (p_mm[9] - p_mm[0]).norm()
    assert torch.isclose(step, torch.tensor(2 * PHYSICAL_REF_MM / 8), atol=1e-3)

    # Legacy geometry on the same volume shears the tilted direction (distance differs)
    legacy = VolumeReslicer(slice_size=(9, 9), rotation_mode="axis_angle")
    g_leg = legacy.compute_grid(torch.tensor([[0.0, 0.0, 0.0, angle, 0.0, 0.0]]), shape)[0, 0].reshape(-1, 3)
    p_leg = g_leg * torch.tensor(half)
    assert abs((p_leg[9] - p_leg[0]).norm().item() - (p_leg[1] - p_leg[0]).norm().item()) > 0.5

    with pytest.raises(ValueError):
        reslicer.compute_grid(pose, shape)


def test_physical_mm_translation_samples_correct_elevation():
    """A pure 10 mm elevational translation must sample the plane z = +10 mm in a z-ramp volume of any spacing."""
    from src.utils.reslicer import PHYSICAL_REF_MM
    d, h, w, dz = 41, 32, 32, 2.0
    z_mm = (torch.arange(d, dtype=torch.float32) - (d - 1) / 2.0) * dz
    vol = z_mm.view(1, 1, d, 1, 1).expand(1, 1, d, h, w).contiguous()
    half = ((w - 1) * 1.0 / 2, (h - 1) * 1.0 / 2, (d - 1) * dz / 2)
    reslicer = VolumeReslicer(slice_size=(16, 16), rotation_mode="axis_angle", ref_mm=PHYSICAL_REF_MM)
    pose = torch.tensor([[0.0, 0.0, 10.0 / PHYSICAL_REF_MM, 0.0, 0.0, 0.0]])
    sl = reslicer(vol, pose, half_extent_mm=half)
    inside = sl[0, 0, 7:9, 7:9]
    assert torch.allclose(inside, torch.full_like(inside, 10.0), atol=1e-3)


def test_reslice_alias_forwards_half_extent_in_physical_mode():
    """reslice() must pass half_extent_mm through to forward() and match it exactly in physical-mm geometry."""
    from src.utils.reslicer import PHYSICAL_REF_MM
    torch.manual_seed(0)
    vol = torch.rand(1, 1, 24, 20, 16)
    half = ((16 - 1) / 2.0, (20 - 1) / 2.0, (24 - 1) / 2.0)
    reslicer = VolumeReslicer(slice_size=(12, 12), rotation_mode="axis_angle", ref_mm=PHYSICAL_REF_MM)
    pose = torch.tensor([[0.02, -0.01, 0.03, 0.1, -0.2, 0.05]])
    assert torch.equal(reslicer.reslice(vol, pose, half_extent_mm=half), reslicer(vol, pose, half_extent_mm=half))
    with pytest.raises(ValueError):
        reslicer.reslice(vol, pose)
