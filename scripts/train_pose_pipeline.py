"""
Single-Volume 6D Pose Regression Pipeline Training Loop (Sprint 4, Task 4.4).

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Implements the continuous 6D pose estimation training loop on a single patient volume (Case 1):
1. Loads preoperative 3D MRI (Case1-FLAIR.nii.gz).
2. Generates on-the-fly random 6D planar poses within clinical bounds:
   - Translation offsets Delta t in [-20, 20] mm (normalized to [-1, 1]).
   - Continuous rotations theta in [-30 deg, +30 deg].
3. Virtually reslices the 3D volume using VolumeReslicer to obtain clean 2D MRI slices.
4. Degrades clean MRI slices via AcousticSimulator to synthesize paired intraoperative ultrasound slices.
5. Passes pairs through NN1 (MRIEncoder), NN2 (USEncoder), and NN3 (PoseRegressor).
6. Computes the composite MultiModalPoseLoss:
   L_total = modality_loss + 1.0 * (translation_loss + rotation_loss).
   The translation residual is scaled per volume by volume_half_extent_mm / trans_bound_mm so that a
   20 mm error equals 1.0 on every axis, and --contrastive_weight adds an in-batch InfoNCE term that
   repels non-matching slices (prevents US embedding collapse).
7. Optimizes all networks via AdamW with cosine annealing and gradient clipping.
8. Tracks training dynamics and exports visual training curves to docs/sprint_4_initial_training_curves.png.
9. Every --log_every epochs, reports true-millimetre MTE on a fixed seeded set from the withheld
   patient through both NN1 -> NN3 (MRI) and NN2 -> NN3 (simulated US). Monitoring only: the saved
   checkpoint is chosen on training loss, never on the withheld patient.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def __len__(self):
            return len(self.iterable) if hasattr(self.iterable, "__len__") else 0
        def set_postfix(self, *args, **kwargs):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass

from scripts.train_contrastive import get_loso_splits
from src.losses.pose_loss import MultiModalPoseLoss
from src.models.encoders import NN1, NN2, MRIEncoder, USEncoder
from src.models.pose_regressor import NN3, PoseRegressor
from src.simulator.sim_us import AcousticSimulator
from src.utils.dataset import load_canonical_nifti, normalize_mri
from src.utils.metrics import compute_mae, compute_mte, pose_unit_mm, volume_half_extent_mm
from src.utils.reslicer import (
    PHYSICAL_REF_MM,
    VolumeReslicer,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    physical_to_normalized_pose,
)


def save_diagnostic_grid(
    epoch: int,
    mri_slice: torch.Tensor,
    us_slice: torch.Tensor,
    gt_plane: torch.Tensor,
    pred_plane: torch.Tensor,
    mte_mm: float,
    mae_deg: float,
    output_dir: Path,
) -> Path:
    """Save 4-panel diagnostic visual comparison grid to track alignment convergence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"val_epoch_{epoch}_visual.png"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.suptitle(f"FCN05 Dynamic Validation Alignment — Epoch {epoch}", fontsize=13, fontweight="bold", y=0.98)

    # 1. Clean Input MRI Slice
    axes[0].imshow(mri_slice[0, 0].detach().cpu().numpy(), cmap="gray", origin="lower")
    axes[0].set_title("1) Input Clean MRI Slice", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. Synthetic Ultrasound Slice
    axes[1].imshow(us_slice[0, 0].detach().cpu().numpy(), cmap="gray", origin="lower")
    axes[1].set_title("2) Synthetic US (Sim Input)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # 3. Ground Truth Resliced MRI Plane
    axes[2].imshow(gt_plane[0, 0].detach().cpu().numpy(), cmap="gray", origin="lower")
    axes[2].set_title("3) Ground-Truth Plane (p_gt)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # 4. Predicted Resliced Plane
    color = "#c53030" if mte_mm > 3.0 else "#22543d"
    axes[3].imshow(pred_plane[0, 0].detach().cpu().numpy(), cmap="gray", origin="lower")
    axes[3].set_title(f"4) Predicted Plane (p_pred)\nMTE: {mte_mm:.2f}mm | MAE: {mae_deg:.1f}°",
                      fontsize=10, fontweight="bold", color=color)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def find_case_volume(case_name: str, base_dir: Path) -> Path:
    """Locate the preoperative FLAIR or T1 MRI volume for any patient case."""
    search_dirs = [
        base_dir / case_name / "MRI",
        base_dir / case_name,
        Path("data/resect_base/RESECT/NIFTI") / case_name / "MRI",
        Path("data/resect_base/RESECT/MINC") / case_name / "MRI",
        Path("data/jv8bk-osfstorage-archive/RESECT-Segmentation") / case_name,
    ]
    for cdir in search_dirs:
        if cdir.exists():
            cands = (
                list(cdir.glob("*FLAIR*.nii.gz"))
                or list(cdir.glob("*FLAIR*.mnc"))
                or list(cdir.glob("*T1*.nii.gz"))
                or list(cdir.glob("*.nii.gz"))
                or list(cdir.glob("*.mnc"))
            )
            if cands:
                return cands[0]
    return base_dir / case_name / "MRI" / f"{case_name}-FLAIR.nii.gz"


def load_or_create_volume(
    data_path: Path,
    device: torch.device,
    canonical: bool = True,
    plane: str = "sagittal",
    allow_synthetic: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Load an MRI volume from disk (a file, or a directory searched for a FLAIR/T1 NIfTI or MINC file).

    Real-data runs must never be evaluated on a substitute volume, so any failure to find or read the volume
    (FileNotFoundError, MemoryError, OSError, a corrupt file, ...) is raised. A synthetic anatomical phantom is
    returned only when allow_synthetic=True and no volume file is found, which unit tests and dry runs request
    explicitly; a volume file that exists but cannot be read is always an error.

    Returns:
        vol_tensor: [1, 1, D, H, W] normalized float32 tensor on target device.
        vol_shape: (D, H, W) integer tuple.
        spacing: (sz, sy, sx) physical voxel spacing in mm.
    """
    target_file = Path(data_path) if data_path is not None else None
    if target_file is not None and target_file.is_dir():
        cands = (
            list(target_file.glob("**/*FLAIR*.nii.gz"))
            or list(target_file.glob("**/*FLAIR*.mnc"))
            or list(target_file.glob("**/*T1*.nii.gz"))
            or list(target_file.glob("**/*.nii.gz"))
        )
        if cands:
            target_file = cands[0]

    if target_file is not None and target_file.is_file():
        print(f"[*] Ingesting clinical volume from: {target_file}")
        # canonical=True: reoriented to RAS+ so that every patient shares the same anatomical axes
        # (Cases 4 and 11 are stored PIR / ASR); canonical=False keeps the stored array order (Sprint 7-8b)
        if canonical:
            raw_data, zooms, _ = load_canonical_nifti(target_file, plane=plane)
        else:
            import nibabel as nib
            img = nib.load(str(target_file))
            raw_data, zooms = img.get_fdata(dtype=np.float32), img.header.get_zooms()[:3]
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        norm_data = normalize_mri(raw_data)
        vol_tensor = torch.from_numpy(norm_data).unsqueeze(0).unsqueeze(0).to(device)
        d, h, w = vol_tensor.shape[-3:]
        print(f"    Loaded volume with shape {d}x{h}x{w}, spacing: {spacing} mm")
        return vol_tensor, (d, h, w), spacing

    if not allow_synthetic:
        raise FileNotFoundError(
            f"No MRI volume found at {data_path}. Real-data runs never substitute a synthetic phantom; "
            "pass allow_synthetic=True (tests and dry runs only) to generate one.")

    # High-contrast synthetic neuroanatomical phantom (explicitly requested)
    print("[*] Generating synthetic neuroanatomical phantom volume (allow_synthetic=True)...")
    d, h, w = 128, 128, 128
    spacing = (1.0, 1.0, 1.0)
    vol = np.zeros((d, h, w), dtype=np.float32)

    # Ellipsoidal brain parenchymal boundary
    z_grid, y_grid, x_grid = np.ogrid[:d, :h, :w]
    cz, cy, cx = d // 2, h // 2, w // 2
    brain_mask = ((z_grid - cz) ** 2 / (0.42 * d) ** 2 +
                  (y_grid - cy) ** 2 / (0.44 * h) ** 2 +
                  (x_grid - cx) ** 2 / (0.38 * w) ** 2) <= 1.0
    vol[brain_mask] = 0.65

    # Ventricular anechoic cavity (dark signal)
    ventricle_mask = ((z_grid - cz) ** 2 / (0.15 * d) ** 2 +
                      (y_grid - cy) ** 2 / (0.08 * h) ** 2 +
                      (x_grid - cx) ** 2 / (0.12 * w) ** 2) <= 1.0
    vol[ventricle_mask] = 0.05

    # Hyperintense lesion/tumor boundary
    tumor_mask = ((z_grid - (cz + 15)) ** 2 / (0.10 * d) ** 2 +
                  (y_grid - (cy - 12)) ** 2 / (0.10 * h) ** 2 +
                  (x_grid - (cx + 10)) ** 2 / (0.10 * w) ** 2) <= 1.0
    vol[tumor_mask] = 0.95

    # Add anatomical gyral texture noise
    rng = np.random.RandomState(42)
    vol += rng.normal(0.0, 0.04, size=vol.shape).astype(np.float32) * brain_mask
    vol = np.clip(vol, 0.0, 1.0)

    vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    return vol_tensor, (d, h, w), spacing


def sample_random_6d_poses(
    batch_size: int,
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    trans_bound_mm: float = 20.0,
    rot_bound_deg: float = 30.0,
    device: torch.device = torch.device("cpu"),
    ref_mm: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample random physical poses and convert to 9D pose vector (3 translation, 6 rotation).

    Args:
        batch_size: Number of poses to sample.
        volume_shape: (D, H, W) of the volume.
        spacing: (sz, sy, sx) physical spacing in mm.
        trans_bound_mm: Maximum translation perturbation (+/- mm).
        rot_bound_deg: Maximum rotation perturbation (+/- degrees).
        device: Target torch device.

    Returns:
        p_gt: Tensor of shape [B, 9] containing [tx_norm, ty_norm, tz_norm, a1_xyz, a2_xyz].
    """
    # 1. Random physical translation in mm [-bound, +bound]
    t_mm = (torch.rand(batch_size, 3, device=device) * 2.0 - 1.0) * trans_bound_mm
    dummy_pose_mm = torch.cat([t_mm, torch.zeros(batch_size, 3, device=device)], dim=-1)

    # 2. Normalize translations to [-1, 1] relative to volume geometry
    if ref_mm is not None:
        # Physical geometry (VolumeReslicer(ref_mm=...)): translation in units of ref_mm on every axis
        t_norm = t_mm / ref_mm
    else:
        t_norm = physical_to_normalized_pose(dummy_pose_mm, volume_shape, spacing)[..., :3]

    # 3. Random rotation axis and bounded angle
    axis = torch.randn(batch_size, 3, device=device)
    axis = F.normalize(axis, p=2, dim=-1, eps=1e-8)
    max_angle_rad = float(rot_bound_deg * math.pi / 180.0)
    angles = (torch.rand(batch_size, 1, device=device) * 2.0 - 1.0) * max_angle_rad

    # 4. Compute 3x3 rotation matrix and extract continuous 6D rotation
    r_axis_angle = axis * angles
    R = axis_angle_to_matrix(r_axis_angle)
    d6 = matrix_to_rotation_6d(R)

    # 5. Composite 9D ground truth vector
    p_gt = torch.cat([t_norm, d6], dim=-1)
    return p_gt


def train_pose_pipeline(
    data_path: Optional[Path] = None,
    epochs: int = 15,
    steps_per_epoch: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    trans_bound_mm: float = 20.0,
    rot_bound_deg: float = 30.0,
    train_translation_only: bool = False,
    train_rotation_only: bool = False,
    vis_freq: int = 1,
    dry_run: bool = False,
    output_plot: Path = Path("docs/sprint_4_initial_training_curves.png"),
    checkpoint_dir: Path = Path("checkpoints"),
    device_name: Optional[str] = None,
    from_scratch: bool = False,
    val_case: str = "Case23",
    case: Optional[str] = None,
    contrastive_weight: float = 0.0,
    temperature: float = 0.1,
    scale_translation: bool = True,
    pretrained: bool = False,
    amp: bool = False,
    quiet: bool = False,
    log_every: int = 1,
    seed: Optional[int] = None,
    mirror_dirs: Tuple[Path, ...] = (),
    allow_synthetic: bool = False,
    physical_geometry: bool = True,
    legacy_simulator: bool = False,
    tz_weight: float = 1.5,
    rot_curriculum_epochs: int = 0,
    canonical_orientation: bool = True,
    slice_plane: str = "axial",
    symmetric_trunk: bool = True,
) -> Dict[str, List[float]]:
    """Execute the 6D pose estimation training pipeline (supporting full cohort & single case)."""
    t_start = time.time()
    # 1. Device configuration
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")

    # Incremental 'baby steps' modes
    if train_translation_only:
        rot_bound_deg = 0.0
        print("[*] INCREMENTAL MODE: Translation Only (--train_translation_only). Rotation perturbations frozen to 0 deg.")
    elif train_rotation_only:
        trans_bound_mm = 0.0
        print("[*] INCREMENTAL MODE: Rotation Only (--train_rotation_only). Translation perturbations frozen to 0 mm.")
    else:
        print("[*] FULL JOINT MODE: Simultaneous Translation + Rotation Optimization.")

    # Dry-run overrides
    if dry_run:
        epochs = 2
        steps_per_epoch = 5
        batch_size = 2
        print(f"[!] DRY-RUN MODE ENABLED: Running {epochs} epochs with {steps_per_epoch} steps each.")

    # 2. Determine Single-Case or Full-Cohort LOSO Partitioning
    if data_path is None:
        for cand in [
            Path("/content/data/resect_base/RESECT/NIFTI"),
            Path("data/resect_base/RESECT/NIFTI"),
            Path("data/resect_base/RESECT/MINC"),
        ]:
            if cand.exists():
                data_path = cand
                break
        if data_path is None:
            data_path = Path("data/resect_base/RESECT/NIFTI")

    cohort_mode = False
    train_cases: List[str] = []
    val_cases: List[str] = []

    if data_path.is_dir() and (case is None or case.lower() == "all"):
        cohort_mode = True
        train_cases, val_cases = get_loso_splits(data_path, val_case=val_case)
        print(f"[*] MULTI-SUBJECT LOSO MODE: {len(train_cases)} Training Patients, Withheld Val Patient: {val_cases[0]}")
    elif case is not None and case.lower() != "all":
        train_cases = [case]
        val_cases = [case]
        print(f"[*] SINGLE-PATIENT MODE: Dedicated to {case}")
    else:
        # Single volume file
        train_cases = ["Case1"]
        val_cases = ["Case1"]
        print(f"[*] SINGLE-VOLUME FILE MODE: {data_path.name}")

    # Volume cache to ensure fast zero-I/O in-memory reslicing across epochs
    volume_cache: Dict[str, Tuple[torch.Tensor, Tuple[int, int, int], Tuple[float, float, float]]] = {}

    def fetch_volume(cname: str) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[float, float, float]]:
        if cname not in volume_cache:
            if data_path.is_dir():
                vol_file = find_case_volume(cname, data_path)
            else:
                vol_file = data_path
            vol_tensor, vol_shape, spacing = load_or_create_volume(vol_file, device, canonical=canonical_orientation, plane=slice_plane,
                                                                   allow_synthetic=allow_synthetic)
            volume_cache[cname] = (vol_tensor, vol_shape, spacing)
        return volume_cache[cname]

    # Pre-cache validation volume
    val_case_id = val_cases[0] if val_cases else "Case23"
    val_vol_tensor, val_vol_shape, val_spacing = fetch_volume(val_case_id)

    # 3. Initialize pipeline components
    ref_mm = PHYSICAL_REF_MM if physical_geometry else None
    reslicer = VolumeReslicer(slice_size=(227, 227), rotation_mode="rotation_6d", ref_mm=ref_mm).to(device)
    simulator = AcousticSimulator.legacy(device=device) if legacy_simulator else AcousticSimulator(device=device)
    print(f"[*] Geometry: {'physical mm (ref ' + str(ref_mm) + ' mm)' if ref_mm else 'legacy normalised'} | "
          f"simulator: {simulator.config()} | NN3 trunk symmetric: {symmetric_trunk} | Tz weight: {tz_weight}")

    def half_of(shape: Tuple[int, int, int], spacing: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return volume_half_extent_mm(shape, spacing)

    # Fixed seeded monitoring set on the withheld patient (both modalities), built before the
    # training RNG stream is seeded so it is identical across runs
    val_half = pose_unit_mm(val_vol_shape, val_spacing, ref_mm)
    val_extent = half_of(val_vol_shape, val_spacing)
    with torch.no_grad():
        torch.manual_seed(1234)
        mon_p_gt = sample_random_6d_poses(32, val_vol_shape, val_spacing, 20.0, 30.0, device=device, ref_mm=ref_mm)
        mon_mri = reslicer(val_vol_tensor, mon_p_gt, half_extent_mm=val_extent)
        mon_us = simulator(mon_mri)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if pretrained:
        print("[*] Initialising NN1 & NN2 backbones from ImageNet ResNet-18 weights (conv1 averaged to 1 channel).")
    nn1 = NN1(embedding_dim=128, pretrained=pretrained).to(device)
    nn2 = NN2(embedding_dim=128, pretrained=pretrained).to(device)
    nn3 = NN3(in_features=128, symmetric_trunk=symmetric_trunk).to(device)

    # Checkpoint loading logic: bypass if from_scratch is set
    if from_scratch:
        print("[*] TRAINING FROM SCRATCH (--from_scratch): Bypassing contrastive checkpoint. NN1 & NN2 randomly initialized.")
    else:
        contrastive_ckpt = checkpoint_dir / "best_model_contrastive.pth"
        if contrastive_ckpt.exists():
            print(f"[*] Loading pre-trained contrastive weights from {contrastive_ckpt}...")
            try:
                ckpt = torch.load(contrastive_ckpt, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                nn1_dict = {k.replace("encoder_mr.", "backbone.").replace("projector_mr.", "projector."): v
                            for k, v in state_dict.items() if "mr" in k}
                nn2_dict = {k.replace("encoder_us.", "backbone.").replace("projector_us.", "projector."): v
                            for k, v in state_dict.items() if "us" in k}
                if nn1_dict:
                    nn1.load_state_dict(nn1_dict, strict=False)
                if nn2_dict:
                    nn2.load_state_dict(nn2_dict, strict=False)
                print("    Successfully loaded pre-trained encoder weights.")
            except Exception as e:
                print(f"    Notice: Could not load contrastive checkpoint: {e}")
        else:
            init = "ImageNet weights" if pretrained else "scratch"
            print(f"[*] No pre-trained contrastive weights found. Initializing NN1 & NN2 from {init}.")

    # 4. Optimization setup
    all_parameters = list(nn1.parameters()) + list(nn2.parameters()) + list(nn3.parameters())
    optimizer = AdamW(all_parameters, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = MultiModalPoseLoss(
        alpha=1.0,
        beta_rot=1.0,
        train_translation_only=train_translation_only,
        train_rotation_only=train_rotation_only,
        contrastive_weight=contrastive_weight,
        temperature=temperature,
        axis_weights=(1.0, 1.0, tz_weight),
    )
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trans_ref_mm = trans_bound_mm if trans_bound_mm > 0 else 20.0
    print(f"[*] Loss: translation residual scaled per volume (20 mm = 1.0): {scale_translation} | "
          f"contrastive weight {contrastive_weight} (tau={temperature}) | AMP: {use_amp}")

    # History tracking
    history: Dict[str, List[float]] = {
        "total_loss": [],
        "modality_loss": [],
        "contrastive_loss": [],
        "translation_loss": [],
        "rotation_loss": [],
        "val_epoch": [],
        "val_mte_mri_mm": [],
        "val_mte_us_mm": [],
    }

    best_loss = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best_model_pose_pipeline.pth"

    # 5. Training loop
    nn1.train()
    nn2.train()
    nn3.train()

    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} | {'Total Loss':<12} | {'Modality Loss':<14} | {'Trans Loss':<12} | {'Rot Loss':<10}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # Rotation curriculum: the rotation bound grows linearly to rot_bound_deg over the first
        # rot_curriculum_epochs epochs (0 = full range from the start)
        epoch_rot_deg = rot_bound_deg * min(1.0, epoch / rot_curriculum_epochs) if rot_curriculum_epochs > 0 else rot_bound_deg
        epoch_totals = {"total": 0.0, "modality": 0.0, "contrastive": 0.0, "trans": 0.0, "rot": 0.0}

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch [{epoch:02d}/{epochs:02d}] Steps",
            leave=False,
            dynamic_ncols=True,
            disable=quiet,
        )

        # Fresh random patient order every epoch, so no patient is systematically visited twice
        case_order = [train_cases[i] for i in torch.randperm(len(train_cases)).tolist()] if cohort_mode else train_cases

        for step in pbar:
            # Multi-case sampling: cycle through the shuffled training cases across steps
            if cohort_mode:
                active_case = case_order[step % len(case_order)]
            else:
                active_case = train_cases[0]

            curr_vol, curr_shape, curr_spacing = fetch_volume(active_case)

            # A. Sample random 6D poses within bounds
            p_gt = sample_random_6d_poses(
                batch_size=batch_size,
                volume_shape=curr_shape,
                spacing=curr_spacing,
                trans_bound_mm=trans_bound_mm,
                rot_bound_deg=epoch_rot_deg,
                device=device,
                ref_mm=ref_mm,
            )

            # B. Reslice volume to obtain ground truth 2D MRI slices
            mri_slices = reslicer(curr_vol, p_gt, half_extent_mm=half_of(curr_shape, curr_spacing))  # [B, 1, 227, 227]

            # C. Degrade MRI slices to synthetic Ultrasound
            us_slices = simulator(mri_slices)       # [B, 1, 227, 227]

            # D. Forward passes through NN1, NN2, NN3
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                emb1 = nn1(mri_slices)              # [B, 128]
                emb2 = nn2(us_slices)               # [B, 128]

                p_pred_us = nn3(emb2)               # [B, 9]
                p_pred_mri = nn3(emb1)              # [B, 9]

            # E. Compute composite MultiModalPoseLoss (fp32), translation scaled so 20 mm = 1.0
            if scale_translation:
                trans_scale = torch.tensor(pose_unit_mm(curr_shape, curr_spacing, ref_mm), device=device) / trans_ref_mm
            else:
                trans_scale = 1.0
            loss, details = criterion(
                emb1=emb1.float(),
                emb2=emb2.float(),
                p_pred=p_pred_us.float(),
                p_gt=p_gt,
                p_pred_mri=p_pred_mri.float(),
                return_details=True,
                trans_scale=trans_scale,
            )

            # F. Backward pass & step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate step metrics
            epoch_totals["total"] += details["total_loss"]
            epoch_totals["modality"] += details["modality_loss"]
            epoch_totals["contrastive"] += details["contrastive_loss"]
            epoch_totals["trans"] += details["translation_loss"]
            epoch_totals["rot"] += details["rotation_loss"]

            pbar.set_postfix({
                "loss": f"{details['total_loss']:.4f}",
                "mod": f"{details['modality_loss']:.4f}",
                "trans": f"{details['translation_loss']:.4f}",
                "rot": f"{details['rotation_loss']:.4f}",
            })

        scheduler.step()

        # Compute epoch averages
        avg_total = epoch_totals["total"] / steps_per_epoch
        avg_modality = epoch_totals["modality"] / steps_per_epoch
        avg_trans = epoch_totals["trans"] / steps_per_epoch
        avg_rot = epoch_totals["rot"] / steps_per_epoch

        history["total_loss"].append(avg_total)
        history["modality_loss"].append(avg_modality)
        history["contrastive_loss"].append(epoch_totals["contrastive"] / steps_per_epoch)
        history["translation_loss"].append(avg_trans)
        history["rotation_loss"].append(avg_rot)

        log_now = epoch == 1 or epoch % max(log_every, 1) == 0 or epoch == epochs
        if log_now:
            # True-millimetre monitoring on the fixed withheld-patient set, both pathways
            nn1.eval()
            nn2.eval()
            nn3.eval()
            with torch.no_grad():
                mte_mri = float(compute_mte(nn3(nn1(mon_mri)), mon_p_gt, trans_bound_mm=val_half).item())
                mte_us = float(compute_mte(nn3(nn2(mon_us)), mon_p_gt, trans_bound_mm=val_half).item())
            nn1.train()
            nn2.train()
            nn3.train()
            history["val_epoch"].append(epoch)
            history["val_mte_mri_mm"].append(mte_mri)
            history["val_mte_us_mm"].append(mte_us)
            print(f"{epoch:<8} | {avg_total:<12.5f} | {avg_modality:<14.5f} | {avg_trans:<12.5f} | {avg_rot:<10.5f}"
                  f" | {val_case_id} MTE mri {mte_mri:5.2f} / us {mte_us:5.2f} mm | {time.time() - t_start:5.0f} s")

        # Dynamic In-Training Validation & 4-Panel Visualization Hook (evaluated on withheld val patient).
        # Its pose and speckle draws run in a forked RNG state, so the training random stream is identical
        # whether or not (and how often) the hook runs.
        if epoch % vis_freq == 0 or epoch == epochs:
            nn1.eval()
            nn2.eval()
            nn3.eval()
            rng_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
                val_p_gt = sample_random_6d_poses(
                    batch_size=1,
                    volume_shape=val_vol_shape,
                    spacing=val_spacing,
                    trans_bound_mm=20.0,
                    rot_bound_deg=30.0,
                    device=device,
                    ref_mm=ref_mm,
                )
                val_mri = reslicer(val_vol_tensor, val_p_gt, half_extent_mm=val_extent)
                val_us = simulator(val_mri)
                val_emb = nn2(val_us)
                val_p_pred = nn3(val_emb)
                val_pred_plane = reslicer(val_vol_tensor, val_p_pred, half_extent_mm=val_extent)

                val_mte = float(compute_mte(val_p_pred[..., :3], val_p_gt[..., :3], trans_bound_mm=val_half).item())
                val_mae = float(compute_mae(val_p_pred, val_p_gt).item())

                vis_path = save_diagnostic_grid(
                    epoch=epoch,
                    mri_slice=val_mri,
                    us_slice=val_us,
                    gt_plane=val_mri,
                    pred_plane=val_pred_plane,
                    mte_mm=val_mte,
                    mae_deg=val_mae,
                    output_dir=checkpoint_dir,
                )
                if not quiet:
                    print(f"    [*] Validation Visual Exported: {vis_path.name} | MTE: {val_mte:.2f}mm, MAE: {val_mae:.1f}°")
            nn1.train()
            nn2.train()
            nn3.train()

        # Save best model
        # Under the rotation curriculum the loss is only comparable once the full rotation range is in use,
        # so earlier epochs (smaller tilts, artificially low loss) are never selected
        if avg_total < best_loss and not dry_run and epoch >= rot_curriculum_epochs:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "nn1_state_dict": nn1.state_dict(),
                "nn2_state_dict": nn2.state_dict(),
                "nn3_state_dict": nn3.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "history": history,
                "config": {
                    "epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_size": batch_size, "lr": lr,
                    "contrastive_weight": contrastive_weight, "temperature": temperature,
                    "scale_translation": scale_translation, "pretrained": pretrained, "amp": use_amp,
                    "seed": seed, "val_case": val_case_id,
                    "geometry": "physical" if ref_mm else "legacy", "ref_mm": ref_mm,
                    "simulator": simulator.config(), "symmetric_trunk": symmetric_trunk, "tz_weight": tz_weight,
                    "rot_curriculum_epochs": rot_curriculum_epochs,
                    "orientation": "RAS" if canonical_orientation else "native",
                    "slice_plane": slice_plane,
                },
            }, best_ckpt_path)

    print("=" * 70)
    print(f"[*] Training complete in {time.time() - t_start:.0f} s. Final Total Loss: {history['total_loss'][-1]:.5f}")

    # 6. Plot and export training dynamics
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, output_plot)
    for d in mirror_dirs:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_plot, d / output_plot.name)

    return history


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Generate and save publication-grade training dynamics figure (no baked-in title)."""
    epochs = list(range(1, len(history["total_loss"]) + 1))
    has_val = bool(history.get("val_epoch"))

    fig, axes = plt.subplots(2, 3 if has_val else 2, figsize=(16 if has_val else 12, 8.5))
    axes = np.atleast_2d(axes)

    def _panel(ax, ys, style, color, title, ylabel):
        ax.plot(epochs, ys, style, color=color, linewidth=1.8, markersize=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    _panel(axes[0, 0], history["total_loss"], "o-", "#1a365d", r"(a) Total loss $\mathcal{L}_{\mathrm{total}}$", "Loss")
    _panel(axes[0, 1], history["modality_loss"], "s-", "#2b6cb0", r"(b) Modality loss $\mathcal{L}_{\mathrm{modality}}$", "Loss")
    _panel(axes[1, 0], history["translation_loss"], "^-", "#c53030",
           r"(d) Translation loss $\mathcal{L}_{\mathrm{trans}}$ (20 mm = 1)", "Scaled MSE")
    _panel(axes[1, 1], history["rotation_loss"], "d-", "#805ad5",
           r"(e) Rotation loss $\|R_{\mathrm{pred}} - R_{\mathrm{gt}}\|_F^2$", "Chordal Frobenius")

    if has_val:
        ax = axes[0, 2]
        _panel(ax, history.get("contrastive_loss", [0.0] * len(epochs)), "v-", "#2f855a",
               r"(c) Contrastive term (InfoNCE + $1-\cos$)", "Loss")
        ax = axes[1, 2]
        ax.plot(history["val_epoch"], history["val_mte_mri_mm"], "o-", color="#1f4e9c", lw=1.8, ms=4,
                label=r"MRI $\rightarrow$ NN1 $\rightarrow$ NN3")
        ax.plot(history["val_epoch"], history["val_mte_us_mm"], "s-", color="#c0392b", lw=1.8, ms=4,
                label=r"Sim-iUS $\rightarrow$ NN2 $\rightarrow$ NN3")
        ax.set_title("(f) Withheld-patient MTE (monitoring only)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("MTE (mm)", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Training dynamics plot exported to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FCN05 Pose Pipeline Trainer (Single-Volume & Full-Cohort)")
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional patient case identifier (e.g. Case1, Case23, Case27). If omitted, trains across full LOSO cohort.",
    )
    parser.add_argument(
        "--val_case",
        type=str,
        default="Case23",
        help="Patient case identifier withheld for validation under LOSO protocol (default: Case23)",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Bypass contrastive pre-trained weights and train NN1, NN2, NN3 from random initialization.",
    )
    parser.add_argument(
        "--data_path", "--data_dir",
        type=Path,
        dest="data_path",
        default=None,
        help="Path to clinical MRI volume file or root directory containing cases",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="Steps per training epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers (for pipeline compatibility)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--trans_bound_mm", type=float, default=20.0, help="Translation perturbation bound in mm")
    parser.add_argument("--rot_bound_deg", type=float, default=30.0, help="Rotation perturbation bound in degrees")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick 2-epoch dry run")
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=Path("docs/sprint_4_initial_training_curves.png"),
        help="Destination path for training curve figure",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=Path,
        default=Path("checkpoints/best_model_pose_pipeline.pth"),
        help="Checkpoint destination",
    )
    parser.add_argument("--train_translation_only", action="store_true",
                        help="Incremental baby-step: only optimize translation MSE loss (rotation perturbations zeroed)")
    parser.add_argument("--train_rotation_only", action="store_true",
                        help="Incremental baby-step: only optimize rotation Frobenius loss (translation perturbations zeroed)")
    parser.add_argument("--vis_freq", type=int, default=1,
                        help="Epoch frequency to export 4-panel diagnostic validation image grid (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--contrastive_weight", type=float, default=0.0,
                        help="Weight of the in-batch InfoNCE + (1 - cos) modality term (default 0 = legacy MSE only)")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature (default: 0.1)")
    parser.add_argument("--legacy_translation_scale", action="store_true",
                        help="Use the unscaled normalised translation MSE of the Sprint 7 run")
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialise NN1/NN2 backbones from ImageNet ResNet-18 weights")
    parser.add_argument("--amp", action="store_true", help="Mixed-precision (fp16 autocast) encoder forward passes")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--log_every", type=int, default=1, help="Print a summary line every N epochs")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the training RNG stream")
    parser.add_argument("--legacy_geometry", action="store_true",
                        help="Apply rotations in the normalised array frame (Sprint 7/8a); default is physical mm")
    parser.add_argument("--legacy_simulator", action="store_true",
                        help="Sprint 7/8a simulator (no reflection or fan, batch-wide normalisation)")
    parser.add_argument("--tz_weight", type=float, default=1.5, help="Weight of the elevational squared error")
    parser.add_argument("--slice_plane", type=str, default="axial", choices=["sagittal", "axial"],
                        help="Canonical slice plane of the RAS volumes (axial = elevation along inferior-superior)")
    parser.add_argument("--native_orientation", action="store_true",
                        help="Keep each NIfTI in its stored axis order (Sprint 7-8b); default reorients to RAS+")
    parser.add_argument("--rot_curriculum_epochs", type=int, default=0,
                        help="Ramp the rotation bound linearly from 0 to --rot_bound_deg over this many epochs")
    parser.add_argument("--legacy_trunk", action="store_true", help="Sprint 7/8a NN3 trunk (133,897 parameters)")
    parser.add_argument("--mirror_dir", type=Path, action="append", default=[],
                        help="Extra directory to copy the training-curve figure into (repeatable)")
    parser.add_argument("--allow_synthetic", action="store_true",
                        help="Use a synthetic phantom when no MRI volume is found (dry runs only; real runs fail loudly)")

    args = parser.parse_args()

    data_path = args.data_path
    if data_path is None:
        for cand in [
            Path("/content/data/resect_base/RESECT/NIFTI"),
            Path("data/resect_base/RESECT/NIFTI"),
            Path("data/resect_base/RESECT/MINC"),
        ]:
            if cand.exists():
                data_path = cand
                break
        if data_path is None:
            data_path = Path("data/resect_base/RESECT/NIFTI")

    if data_path.is_dir() and args.case is not None:
        data_path = find_case_volume(args.case, data_path)
    elif args.case is not None and not data_path.exists():
        data_path = find_case_volume(args.case, Path("data/resect_base/RESECT/NIFTI"))

    train_pose_pipeline(
        data_path=data_path,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        trans_bound_mm=args.trans_bound_mm,
        rot_bound_deg=args.rot_bound_deg,
        train_translation_only=args.train_translation_only,
        train_rotation_only=args.train_rotation_only,
        vis_freq=args.vis_freq,
        dry_run=args.dry_run,
        output_plot=args.output_plot,
        checkpoint_dir=args.save_checkpoint.parent,
        device_name=args.device,
        from_scratch=args.from_scratch,
        val_case=args.val_case,
        case=args.case,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        scale_translation=not args.legacy_translation_scale,
        pretrained=args.pretrained,
        amp=args.amp,
        quiet=args.quiet,
        log_every=args.log_every,
        seed=args.seed,
        mirror_dirs=tuple(args.mirror_dir),
        allow_synthetic=args.allow_synthetic,
        physical_geometry=not args.legacy_geometry,
        legacy_simulator=args.legacy_simulator,
        tz_weight=args.tz_weight,
        rot_curriculum_epochs=args.rot_curriculum_epochs,
        canonical_orientation=not args.native_orientation,
        slice_plane=args.slice_plane,
        symmetric_trunk=not args.legacy_trunk,
    )


if __name__ == "__main__":
    main()
