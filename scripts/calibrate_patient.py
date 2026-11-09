"""
Pre-Operative Zero-US Patient Calibration Protocol (Sprint 5, Task 5.1; dual-branch revision Sprint 8).

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Simulates the 'night-before-surgery' calibration protocol:
1. Ingests a new patient's high-resolution preoperative 3D MRI (e.g. Case 23).
2. Loads pre-trained pipeline weights (best_model_pose_pipeline.pth) and rebuilds the exact reslicer geometry,
   simulator and NN3 trunk recorded in the checkpoint config (legacy Sprint 7/8a checkpoints have no config).
3. Freezes the ResNet-18 backbones of NN1 (MRIEncoder) and NN2 (USEncoder) (requires_grad = False, eval mode).
4. Fine-tunes either NN3 only (variant 'nn3') or NN3 plus the two 512->128 projection heads ('proj_nn3',
   with the modality loss active); '--variant compare' runs both from the same start and keeps the one with
   the lower pose loss on the selection set.
5. Virtually reslices the patient's 3D MRI across clinical 6D bounds:
   - Translation offsets Delta t in [-20, 20] mm.
   - Continuous rotations theta in [-30 deg, +30 deg].
6. Fine-tunes on BOTH branches (--branches both, default): NN3(NN1(MRI slice)) and NN3(NN2(Sim-iUS(MRI slice))).
   The simulated frames are synthesised from the patient's own MRI, so no real ultrasound is used.
7. Evaluates both pathways on fixed seeded protocol sets in true millimetres:
   - Protocol A: seed 42, 10 x 4 slices, +/-20 mm, +/-15 deg (per-epoch trajectory).
   The saved epoch is chosen on a separate 160-slice selection set (seed 7, +/-30 deg) drawn from the same
   pre-operative MRI, never on the reported protocol sets.
8. Visualizes calibration convergence and saves results to docs/sprint_5_calibration_curves.png.
"""

from __future__ import annotations

import argparse
import copy
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

from scripts.train_pose_pipeline import load_or_create_volume, sample_random_6d_poses
from src.losses.pose_loss import MultiModalPoseLoss
from src.models.encoders import NN1, NN2
from src.models.pose_regressor import NN3, rotation_6d_to_matrix
from src.simulator.sim_us import AcousticSimulator
from src.utils.metrics import compute_mae, compute_mte, pose_unit_mm, volume_half_extent_mm
from src.utils.reslicer import VolumeReslicer

BRANCH_LABELS = {"mri": "MRI -> NN1 -> NN3", "us": "Sim-iUS -> NN2 -> NN3"}


def pipeline_from_config(config: Optional[Dict], device: torch.device):
    """
    Rebuild the data pipeline a checkpoint was trained with.

    Returns (reslicer, simulator, ref_mm, symmetric_trunk, axis_weights, contrastive_weight). A missing
    config means a Sprint 7/8a checkpoint: legacy geometry, legacy simulator, legacy NN3 trunk.
    """
    cfg = config or {}
    ref_mm = cfg.get("ref_mm") if cfg.get("geometry") == "physical" else None
    sim_cfg = cfg.get("simulator")
    if sim_cfg is None:
        simulator = AcousticSimulator.legacy(device=device)
    else:
        simulator = AcousticSimulator(device=device, reflection_weight=sim_cfg["reflection_weight"],
                                      fan_angle_deg=sim_cfg["fan_angle_deg"],
                                      apex_offset_frac=sim_cfg["apex_offset_frac"],
                                      per_slice_norm=sim_cfg["per_slice_norm"],
                                      f0=sim_cfg.get("f0", 5.0), alpha_wm=sim_cfg.get("alpha_wm", 0.452),
                                      sigma_rayleigh=sim_cfg.get("sigma_rayleigh", 59.09))
    reslicer = VolumeReslicer(slice_size=(227, 227), rotation_mode="rotation_6d", ref_mm=ref_mm).to(device)
    axis_weights = (1.0, 1.0, float(cfg["tz_weight"])) if "tz_weight" in cfg else None
    return (reslicer, simulator, ref_mm, bool(cfg.get("symmetric_trunk", False)), axis_weights,
            float(cfg.get("contrastive_weight", 0.0)))


def volume_options(config: Optional[Dict]) -> Dict[str, object]:
    """Volume loading options a checkpoint was trained with (stored orientation and canonical slice plane)."""
    cfg = config or {}
    return {"canonical": cfg.get("orientation", "native") != "native", "plane": cfg.get("slice_plane", "sagittal")}


def freeze_encoders(nn1: nn.Module, nn2: nn.Module) -> Tuple[int, int]:
    """
    Freeze all parameters of NN1 and NN2 to preserve universal modality alignment.

    Args:
        nn1: MRI feature encoder.
        nn2: Ultrasound feature encoder.

    Returns:
        (frozen_nn1_count, frozen_nn2_count): Number of frozen parameters per encoder.
    """
    nn1_params = 0
    for p in nn1.parameters():
        p.requires_grad = False
        nn1_params += p.numel()

    nn2_params = 0
    for p in nn2.parameters():
        p.requires_grad = False
        nn2_params += p.numel()

    nn1.eval()
    nn2.eval()
    return nn1_params, nn2_params


def unfreeze_regressor(nn3: nn.Module) -> int:
    """
    Ensure all parameters of NN3 (PoseRegressor) are unfrozen for patient specialization.

    Args:
        nn3: 6D pose regressor.

    Returns:
        trainable_params_count: Number of trainable parameters in NN3.
    """
    nn3_params = 0
    for p in nn3.parameters():
        p.requires_grad = True
        nn3_params += p.numel()

    nn3.train()
    return nn3_params


def rot_mat_to_euler_xyz(r_rel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose relative rotations into absolute Tait-Bryan XYZ (pitch, roll, yaw) errors in degrees."""
    sy = torch.sqrt(r_rel[..., 0, 0] ** 2 + r_rel[..., 1, 0] ** 2)
    pitch = torch.atan2(r_rel[..., 2, 1], r_rel[..., 2, 2])
    roll = torch.atan2(-r_rel[..., 2, 0], sy)
    yaw = torch.atan2(r_rel[..., 1, 0], r_rel[..., 0, 0])
    return torch.rad2deg(torch.abs(pitch)), torch.rad2deg(torch.abs(roll)), torch.rad2deg(torch.abs(yaw))


def build_eval_set(
    reslicer: VolumeReslicer,
    vol_tensor: torch.Tensor,
    vol_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    n_steps: int = 10,
    batch_size: int = 4,
    trans_bound_mm: float = 20.0,
    rot_bound_deg: float = 30.0,
    seed: Optional[int] = 42,
    simulator: Optional[AcousticSimulator] = None,
    sim_seed: Optional[int] = 1234,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Build a fixed evaluation set of n_steps x batch_size poses, MRI reslices and (optionally) simulated
    US frames. Poses are drawn in batches of batch_size after torch.manual_seed(seed), which reproduces
    the legacy Protocol B poses for seed 42; the speckle stream is then seeded with sim_seed and frames
    are simulated per batch. The pose units follow the reslicer geometry (reslicer.ref_mm).
    """
    ref_mm = getattr(reslicer, "ref_mm", None)
    extent = volume_half_extent_mm(vol_shape, spacing)
    poses: List[torch.Tensor] = []
    mri: List[torch.Tensor] = []
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        for _ in range(n_steps):
            p_gt = sample_random_6d_poses(batch_size, vol_shape, spacing, trans_bound_mm, rot_bound_deg,
                                          device=device, ref_mm=ref_mm)
            poses.append(p_gt)
            mri.append(reslicer(vol_tensor, p_gt, half_extent_mm=extent))
        out = {"p_gt": torch.cat(poses), "mri": torch.cat(mri)}
        if simulator is not None:
            if sim_seed is not None:
                torch.manual_seed(sim_seed)
            out["us"] = torch.cat([simulator(m) for m in mri])
    return out


def per_slice_errors(
    p_pred: torch.Tensor,
    p_gt: torch.Tensor,
    unit_mm: Tuple[float, float, float],
) -> Dict[str, np.ndarray]:
    """Per-slice absolute translation errors (mm), MTE (mm), geodesic and Euler-axis errors (deg)."""
    scale = torch.tensor(unit_mm, device=p_pred.device)
    d_t = torch.abs(p_pred[:, :3] - p_gt[:, :3]) * scale
    mte = compute_mte(p_pred, p_gt, trans_bound_mm=unit_mm, reduction="none")
    geo = compute_mae(p_pred, p_gt, reduction="none")
    r_rel = rotation_6d_to_matrix(p_pred[:, 3:9]).transpose(-1, -2) @ rotation_6d_to_matrix(p_gt[:, 3:9])
    pitch, roll, yaw = rot_mat_to_euler_xyz(r_rel)
    out = {"tx": d_t[:, 0], "ty": d_t[:, 1], "tz": d_t[:, 2], "mte": mte, "geo": geo,
           "pitch": pitch, "roll": roll, "yaw": yaw}
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def summarise_errors(errs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Mean of every per-slice error array, plus the legacy key names."""
    out = {k: float(np.mean(v)) for k, v in errs.items()}
    out.update({"mte_mm": out["mte"], "mae_deg": out["geo"],
                "tx_err_mm": out["tx"], "ty_err_mm": out["ty"], "tz_err_mm": out["tz"]})
    return out


def predict_branch(
    branch: str, eval_set: Dict[str, torch.Tensor], nn1: nn.Module, nn2: Optional[nn.Module], nn3: nn.Module
) -> torch.Tensor:
    """Run one pathway over a fixed evaluation set: 'mri' = NN1 -> NN3, 'us' = NN2 -> NN3."""
    with torch.no_grad():
        if branch == "mri":
            return nn3(nn1(eval_set["mri"]))
        return nn3(nn2(eval_set["us"]))


def evaluate_patient_calibration(
    reslicer: VolumeReslicer,
    nn1: nn.Module,
    nn3: nn.Module,
    vol_tensor: torch.Tensor,
    vol_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    val_steps: int = 10,
    batch_size: int = 4,
    trans_bound_mm: float = 20.0,
    rot_bound_deg: float = 30.0,
    device: torch.device = torch.device("cpu"),
    nn2: Optional[nn.Module] = None,
    eval_set: Optional[Dict[str, torch.Tensor]] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate 6D pose estimation metrics on virtual slices from the patient MRI, in true millimetres.

    Args:
        reslicer: Differentiable 3D volume reslicer.
        nn1: Frozen MRI encoder.
        nn3: Pose regressor.
        vol_tensor: [1, 1, D, H, W] patient volume tensor.
        vol_shape: (D, H, W) tuple.
        spacing: (sz, sy, sx) physical voxel spacing in mm.
        val_steps: Number of evaluation mini-batches (when eval_set is None).
        batch_size: Slices per validation step (when eval_set is None).
        trans_bound_mm: Physical translation perturbation bound in mm.
        rot_bound_deg: Angular perturbation bound in degrees.
        device: Target torch device.
        nn2: Optional frozen US encoder; if given with an eval_set holding 'us', the US branch is reported
             under 'us_*' keys.
        eval_set: Optional fixed set from build_eval_set (otherwise a fresh set is drawn).
        seed: Seed for a freshly drawn set (None = current RNG state).

    Returns:
        Dictionary with aggregated 'mte_mm', 'mae_deg', and axis errors for the MRI branch
        (plus 'us_*' keys for the US branch when available).
    """
    nn1.eval()
    nn3.eval()
    if eval_set is None:
        eval_set = build_eval_set(reslicer, vol_tensor, vol_shape, spacing, val_steps, batch_size,
                                  trans_bound_mm, rot_bound_deg, seed=seed, device=device)
    unit = pose_unit_mm(vol_shape, spacing, getattr(reslicer, "ref_mm", None))

    metrics = summarise_errors(per_slice_errors(predict_branch("mri", eval_set, nn1, nn2, nn3), eval_set["p_gt"], unit))
    if nn2 is not None and "us" in eval_set:
        nn2.eval()
        us = summarise_errors(per_slice_errors(predict_branch("us", eval_set, nn1, nn2, nn3), eval_set["p_gt"], unit))
        metrics.update({f"us_{k}": v for k, v in us.items()})
    return metrics


def reestimate_batchnorm(encoder: nn.Module, images: torch.Tensor, batch_size: int = 16) -> int:
    """
    Re-estimate the BatchNorm running statistics of a frozen encoder on the target patient's own data.

    Stage 1 draws every mini-batch from a single patient, so BatchNorm learns to normalise with that patient's
    statistics, whereas the stored running statistics average over 22 patients. This replaces them with a
    cumulative average of the batch statistics over `images` (no gradients; weights unchanged).

    Returns:
        Number of BatchNorm layers re-estimated.
    """
    bns = [m for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative moving average over the pre-operative batches
    encoder.train()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            encoder(images[i:i + batch_size])
    encoder.eval()
    return len(bns)


def selection_loss(
    criterion: MultiModalPoseLoss,
    sel_set: Dict[str, torch.Tensor],
    branches: str,
    nn1: nn.Module,
    nn2: nn.Module,
    nn3: nn.Module,
    trans_scale: torch.Tensor,
) -> float:
    """Pose loss of the calibrated branches on the held-out pre-operative selection set."""
    nn3.eval()
    with torch.no_grad():
        p_mri = nn3(nn1(sel_set["mri"])) if branches in ("mri", "both") else None
        p_us = nn3(nn2(sel_set["us"])) if branches in ("us", "both") else None
        p_main = p_us if p_us is not None else p_mri
        p_aux = p_mri if p_us is not None else None
        loss = criterion(p_pred=p_main, p_gt=sel_set["p_gt"], p_pred_mri=p_aux, trans_scale=trans_scale)
    return float(loss.item())


def find_patient_volume(
    case_name: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Locate the preoperative FLAIR/T1 MRI volume for any patient case (Case1 to Case27)."""
    search_dirs: List[Path] = []
    if base_dir is not None:
        search_dirs.append(Path(base_dir))
    search_dirs.extend([
        Path("data/resect_base/RESECT/NIFTI"),
        Path("data/resect_base/RESECT/MINC"),
        Path("data/jv8bk-osfstorage-archive/RESECT-Segmentation"),
    ])

    for bdir in search_dirs:
        case_dir = bdir / case_name / "MRI"
        if not case_dir.exists():
            case_dir = bdir / case_name

        if case_dir.exists():
            flair_candidates = list(case_dir.glob("*FLAIR*.nii.gz")) or list(case_dir.glob("*FLAIR*.mnc"))
            if flair_candidates:
                return flair_candidates[0]
            t1_candidates = list(case_dir.glob("*T1*.nii.gz")) or list(case_dir.glob("*T1*.mnc"))
            if t1_candidates:
                return t1_candidates[0]
            nii_candidates = list(case_dir.glob("*.nii.gz")) or list(case_dir.glob("*.mnc"))
            if nii_candidates:
                return nii_candidates[0]

    # Return default expected path
    return Path("data/resect_base/RESECT/NIFTI") / case_name / "MRI" / f"{case_name}-FLAIR.nii.gz"


def calibrate_patient(
    case_name: str = "Case23",
    data_path: Optional[Path] = None,
    checkpoint_path: Path = Path("checkpoints/best_model_pose_pipeline.pth"),
    epochs: int = 5,
    steps_per_epoch: int = 20,
    val_steps: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    trans_bound_mm: float = 20.0,
    rot_bound_deg: float = 30.0,
    dry_run: bool = False,
    output_plot: Path = Path("docs/sprint_5_calibration_curves.png"),
    save_checkpoint: Optional[Path] = None,
    device_name: Optional[str] = None,
    branches: str = "both",
    eval_rot_bound_deg: float = 15.0,
    eval_batch_size: int = 4,
    sel_steps: int = 40,
    seed: Optional[int] = 0,
    quiet: bool = False,
    mirror_dirs: Tuple[Path, ...] = (),
    allow_synthetic: bool = False,
    variant: str = "nn3",
    bn_reestimate: bool = False,
) -> Dict[str, List[float]]:
    """
    Execute the patient-specific zero-US calibration pipeline.

    Simulates the clinical workflow the night before neurosurgery:
    - Pre-operative MRI is loaded.
    - The NN1/NN2 backbones are frozen; NN3 (and, for variant 'proj_nn3', the projection heads) is fine-tuned on
      virtual resliced MRI cuts and on simulated US synthesised from them.
    - Validation MTE (mm) and MAE (deg) are measured on both pathways before and after calibration.
    """
    if branches not in ("mri", "us", "both"):
        raise ValueError(f"branches must be 'mri', 'us' or 'both', got {branches}")
    if variant not in ("nn3", "proj_nn3", "compare"):
        raise ValueError(f"variant must be 'nn3', 'proj_nn3' or 'compare', got {variant}")

    # 1. Device configuration
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Zero-US calibration for {case_name} on {device} (branches: {branches}, variant: {variant})")

    # Dry-run adjustments
    if dry_run:
        epochs = 2
        steps_per_epoch = 5
        val_steps = 2
        batch_size = 2
        eval_batch_size = 2
        print(f"[!] DRY-RUN MODE: {epochs} calibration epochs, {steps_per_epoch} steps/epoch, {val_steps} val steps.")

    # 2. Determine volume path and load volume
    if data_path is None or data_path.is_dir():
        data_path = find_patient_volume(case_name, base_dir=data_path)
    # 3. Load the trained checkpoint, then the volume in the orientation and slice plane it was trained with
    ckpt: Dict = {}
    if checkpoint_path.exists():
        print(f"[*] Ingesting pre-trained checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    elif dry_run:
        print(f"[*] Notice: Checkpoint {checkpoint_path} not found. Dry run: proceeding with initialised weights.")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found; a real calibration needs the trained Stage 1 model.")
    vol_tensor, vol_shape, spacing = load_or_create_volume(data_path, device, allow_synthetic=allow_synthetic,
                                                           **volume_options(ckpt.get("config")))
    extent = volume_half_extent_mm(vol_shape, spacing)
    reslicer, simulator, ref_mm, symmetric, axis_weights, contrastive_weight = pipeline_from_config(ckpt.get("config"), device)
    unit = pose_unit_mm(vol_shape, spacing, ref_mm)
    trans_scale = torch.tensor(unit, device=device) / (trans_bound_mm if trans_bound_mm > 0 else 20.0)

    nn1 = NN1(embedding_dim=128).to(device)
    nn2 = NN2(embedding_dim=128).to(device)
    nn3 = NN3(in_features=128, symmetric_trunk=symmetric).to(device)
    for net, key in ((nn1, "nn1_state_dict"), (nn2, "nn2_state_dict"), (nn3, "nn3_state_dict")):
        if key in ckpt:
            net.load_state_dict(ckpt[key])

    # 4. Freeze the encoders (backbones always stay frozen and in eval mode)
    frozen_nn1, frozen_nn2 = freeze_encoders(nn1, nn2)
    trainable_nn3 = unfreeze_regressor(nn3)
    assert all(not p.requires_grad for p in nn1.parameters()), "NN1 parameters must be frozen!"
    assert all(not p.requires_grad for p in nn2.parameters()), "NN2 parameters must be frozen!"
    assert all(p.requires_grad for p in nn3.parameters()), "NN3 parameters must be trainable!"
    proj_params = sum(p.numel() for p in nn1.projector.parameters()) + sum(p.numel() for p in nn2.projector.parameters())
    print(f"[*] Frozen: NN1 {frozen_nn1:,} + NN2 {frozen_nn2:,} | NN3 {trainable_nn3:,} trainable "
          f"(+{proj_params:,} projector parameters in variant proj_nn3)")

    # 5. Fixed evaluation sets (built before the calibration RNG stream is seeded)
    protocol_a = build_eval_set(reslicer, vol_tensor, vol_shape, spacing, val_steps, eval_batch_size,
                                trans_bound_mm, eval_rot_bound_deg, seed=42, simulator=simulator,
                                sim_seed=1234, device=device)
    sel_set = build_eval_set(reslicer, vol_tensor, vol_shape, spacing, val_steps if dry_run else sel_steps,
                             eval_batch_size, trans_bound_mm, rot_bound_deg, seed=7, simulator=simulator,
                             sim_seed=4321, device=device)
    pose_crit = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, trans_scale=trans_scale, axis_weights=axis_weights)
    full_crit = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, trans_scale=trans_scale, axis_weights=axis_weights,
                                   contrastive_weight=contrastive_weight)
    if bn_reestimate:
        # Zero-US BatchNorm re-estimation on a separate pre-operative set (seed 11): MRI reslices for NN1 and their
        # simulated frames for NN2. Done once, before any fine-tuning, so the readout is calibrated on the new statistics.
        bn_set = build_eval_set(reslicer, vol_tensor, vol_shape, spacing, val_steps if dry_run else sel_steps,
                                eval_batch_size, trans_bound_mm, rot_bound_deg, seed=11, simulator=simulator,
                                sim_seed=5678, device=device)
        n_bn = reestimate_batchnorm(nn1, bn_set["mri"]) + reestimate_batchnorm(nn2, bn_set["us"])
        print(f"[*] BatchNorm statistics re-estimated on {len(bn_set['p_gt'])} pre-operative slices ({n_bn} layers)")
    init_state = {k: copy.deepcopy(n.state_dict()) for k, n in (("nn1", nn1), ("nn2", nn2), ("nn3", nn3))}

    def run_variant(name: str) -> Dict:
        """Calibrate one variant from the shared starting weights; return its history and best weights."""
        for k, n in (("nn1", nn1), ("nn2", nn2), ("nn3", nn3)):
            n.load_state_dict(init_state[k])
        train_proj = name == "proj_nn3"
        for proj in (nn1.projector, nn2.projector):
            for p in proj.parameters():
                p.requires_grad = train_proj
        params = list(nn3.parameters()) + (list(nn1.projector.parameters()) + list(nn2.projector.parameters()) if train_proj else [])
        if seed is not None:
            torch.manual_seed(seed)
        optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        def evaluate() -> Dict[str, float]:
            return evaluate_patient_calibration(reslicer, nn1, nn3, vol_tensor, vol_shape, spacing,
                                                device=device, nn2=nn2, eval_set=protocol_a)

        base = evaluate()
        best_sel = selection_loss(pose_crit, sel_set, branches, nn1, nn2, nn3, trans_scale)
        best_epoch, best = 0, {k: copy.deepcopy(n.state_dict()) for k, n in (("nn1", nn1), ("nn2", nn2), ("nn3", nn3))}
        hist: Dict[str, List[float]] = {"epoch": [0], "train_loss": [float("nan")], "selection_loss": [best_sel]}
        for key in ("mte_mm", "mae_deg", "tx_err_mm", "ty_err_mm", "tz_err_mm"):
            hist[f"val_{key}"] = [base[key]]
            hist[f"val_us_{key}"] = [base[f"us_{key}"]]
        print(f"  [{name}] {'Epoch':<6} | {'Train':<7} | {'Select':<7} | MRI MTE / MAE  | US MTE / MAE")
        print(f"  [{name}] {0:<6} | {'---':<7} | {best_sel:<7.4f} | {base['mte_mm']:5.2f} / {base['mae_deg']:5.2f} "
              f"| {base['us_mte_mm']:5.2f} / {base['us_mae_deg']:5.2f}")

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            nn3.train()
            epoch_loss = 0.0
            for _ in tqdm(range(steps_per_epoch), leave=False, disable=quiet):
                p_gt = sample_random_6d_poses(batch_size, vol_shape, spacing, trans_bound_mm, rot_bound_deg,
                                              device=device, ref_mm=ref_mm)
                mri_slices = reslicer(vol_tensor, p_gt, half_extent_mm=extent)
                with torch.no_grad():
                    f1 = nn1.backbone(mri_slices) if branches in ("mri", "both") else None
                    f2 = nn2.backbone(simulator(mri_slices)) if branches in ("us", "both") else None
                emb1 = F.normalize(nn1.projector(f1), dim=-1) if f1 is not None else None
                emb2 = F.normalize(nn2.projector(f2), dim=-1) if f2 is not None else None
                if branches == "both":
                    crit = full_crit if train_proj else pose_crit
                    loss, details = crit(emb1=emb1 if train_proj else None, emb2=emb2 if train_proj else None,
                                         p_pred=nn3(emb2), p_gt=p_gt, p_pred_mri=nn3(emb1), return_details=True)
                elif branches == "us":
                    loss, details = pose_crit(p_pred=nn3(emb2), p_gt=p_gt, return_details=True)
                else:
                    loss, details = pose_crit(p_pred=nn3(emb1), p_gt=p_gt, return_details=True)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                epoch_loss += details["total_loss"]
            scheduler.step()

            sel = selection_loss(pose_crit, sel_set, branches, nn1, nn2, nn3, trans_scale)
            if sel < best_sel:
                best_sel, best_epoch = sel, epoch
                best = {k: copy.deepcopy(n.state_dict()) for k, n in (("nn1", nn1), ("nn2", nn2), ("nn3", nn3))}
            m = evaluate()
            hist["epoch"].append(epoch)
            hist["train_loss"].append(epoch_loss / steps_per_epoch)
            hist["selection_loss"].append(sel)
            for key in ("mte_mm", "mae_deg", "tx_err_mm", "ty_err_mm", "tz_err_mm"):
                hist[f"val_{key}"].append(m[key])
                hist[f"val_us_{key}"].append(m[f"us_{key}"])
            if epoch == epochs or epoch % 2 == 0 or quiet is False:
                print(f"  [{name}] {epoch:<6} | {epoch_loss / steps_per_epoch:<7.4f} | {sel:<7.4f} | "
                      f"{m['mte_mm']:5.2f} / {m['mae_deg']:5.2f} | {m['us_mte_mm']:5.2f} / {m['us_mae_deg']:5.2f}")
        hist["best_epoch"] = [best_epoch]
        hist["calibration_seconds"] = [time.time() - t0]
        print(f"  [{name}] selected epoch {best_epoch}, selection loss {best_sel:.4f}, {hist['calibration_seconds'][0]:.1f} s")
        return {"history": hist, "best": best, "best_sel": best_sel}

    variants = ["nn3", "proj_nn3"] if variant == "compare" else [variant]
    print("\n" + "=" * 92)
    print(f"PROTOCOL A ({case_name}, seed 42, {len(protocol_a['p_gt'])} slices, +/-{trans_bound_mm:.0f} mm, "
          f"+/-{eval_rot_bound_deg:.0f} deg) | true-mm MTE / geodesic MAE; selection on {len(sel_set['p_gt'])} slices")
    print("=" * 92)
    runs = {name: run_variant(name) for name in variants}
    chosen = min(runs, key=lambda k: runs[k]["best_sel"])
    if len(runs) > 1:
        print("[*] Variant comparison (selection-set pose loss): " +
              ", ".join(f"{k} {v['best_sel']:.4f}" for k, v in runs.items()) + f" -> keeping {chosen}")
    history = runs[chosen]["history"]
    history["variant"] = [chosen]
    history["variant_selection_loss"] = {k: v["best_sel"] for k, v in runs.items()}
    history["variant_histories"] = {k: v["history"] for k, v in runs.items()}
    for k, n in (("nn1", nn1), ("nn2", nn2), ("nn3", nn3)):
        n.load_state_dict(runs[chosen]["best"][k])
    best_epoch = history["best_epoch"][0]

    # 6. Save Calibrated Model Checkpoint (selected variant and epoch)
    if save_checkpoint is None:
        save_checkpoint = Path(f"checkpoints/calibrated_model_{case_name.lower()}.pth")
    if not dry_run:
        save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "case_name": case_name,
            "branches": branches,
            "variant": chosen,
            "nn1_state_dict": nn1.state_dict(),
            "nn2_state_dict": nn2.state_dict(),
            "nn3_state_dict": nn3.state_dict(),
            "best_epoch": best_epoch,
            "baseline_mte_mm": history["val_mte_mm"][0],
            "calibrated_mte_mm": history["val_mte_mm"][best_epoch],
            "baseline_mae_deg": history["val_mae_deg"][0],
            "calibrated_mae_deg": history["val_mae_deg"][best_epoch],
            "history": history,
            "config": {**(ckpt.get("config") or {}),
                       "calibration": {"epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_size": batch_size,
                                       "lr": lr, "trans_bound_mm": trans_bound_mm, "rot_bound_deg": rot_bound_deg,
                                       "eval_rot_bound_deg": eval_rot_bound_deg, "seed": seed, "sel_steps": sel_steps,
                                       "variant": chosen, "bn_reestimate": bn_reestimate}},
        }, save_checkpoint)
        print(f"[*] Calibrated patient model checkpoint saved to: {save_checkpoint}")

    # 7. Generate and Export Calibration Curve Plot
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_calibration_curves(history, case_name, output_plot)
    for d in mirror_dirs:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_plot, d / output_plot.name)

    return history


def plot_calibration_curves(
    history: Dict[str, List[float]],
    case_name: str,
    output_path: Path,
) -> None:
    """Generate and save publication-quality calibration progress plots (both pathways, no baked-in title)."""
    epochs = history["epoch"]
    best = history.get("best_epoch", [None])[0]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, key, ylabel, title in (
        (ax1, "mte_mm", "MTE (mm)", "(a) Mean translation error"),
        (ax2, "mae_deg", "MAE (degrees)", "(b) Mean geodesic angular error"),
    ):
        ax.plot(epochs, history[f"val_{key}"], marker="o", color="#1f4e9c", linewidth=2.2,
                label=r"MRI $\rightarrow$ NN1 $\rightarrow$ NN3")
        if f"val_us_{key}" in history:
            ax.plot(epochs, history[f"val_us_{key}"], marker="s", color="#c0392b", linewidth=2.2,
                    label=r"Sim-iUS $\rightarrow$ NN2 $\rightarrow$ NN3")
        if best is not None:
            ax.axvline(best, color="#555555", linestyle=":", linewidth=1.5, label=f"Selected epoch ({best})")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Calibration epoch (0 = before fine-tuning)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_p), dpi=300)
    plt.close()
    print(f"[*] Calibration curves saved to: {out_p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-US Patient Calibration Benchmark (dual-branch)")
    parser.add_argument("--case", type=str, default="Case23", help="Patient case identifier (default: Case23)")
    parser.add_argument("--data_path", "--data_dir", type=str, dest="data_path", default=None,
                        help="Explicit path to patient NIfTI file or dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_pose_pipeline.pth",
                        help="Path to pre-trained pipeline checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of calibration epochs (default: 5)")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="Steps per epoch (default: 20)")
    parser.add_argument("--val_steps", type=int, default=10, help="Protocol A batches (default: 10)")
    parser.add_argument("--batch_size", type=int, default=4, help="Calibration batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate (default: 1e-4)")
    parser.add_argument("--trans_bound_mm", type=float, default=20.0, help="Translation perturbation bound (+/- mm)")
    parser.add_argument("--rot_bound_deg", type=float, default=30.0, help="Calibration rotation bound (+/- deg)")
    parser.add_argument("--eval_rot_bound_deg", type=float, default=15.0, help="Protocol A rotation bound (+/- deg)")
    parser.add_argument("--branches", type=str, default="both", choices=["mri", "us", "both"],
                        help="Pathways NN3 is calibrated on (default: both)")
    parser.add_argument("--bn_reestimate", action="store_true",
                        help="Re-estimate encoder BatchNorm statistics on the patient's pre-operative data before calibration")
    parser.add_argument("--variant", type=str, default="nn3", choices=["nn3", "proj_nn3", "compare"],
                        help="Trainable set: NN3 only, NN3 + projection heads, or compare both on the selection set")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the calibration RNG stream")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bars and per-epoch lines")
    parser.add_argument("--dry_run", action="store_true", help="Run short verification trial")
    parser.add_argument("--output_plot", type=str, default="docs/sprint_5_calibration_curves.png",
                        help="Path to export calibration curves PNG")
    parser.add_argument("--mirror_dir", type=Path, action="append", default=[],
                        help="Extra directory to copy the calibration figure into (repeatable)")
    parser.add_argument("--allow_synthetic", action="store_true",
                        help="Use a synthetic phantom when no MRI volume is found (dry runs only; real runs fail loudly)")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save calibrated model")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")

    args = parser.parse_args()

    calibrate_patient(
        case_name=args.case,
        data_path=Path(args.data_path) if args.data_path is not None else None,
        checkpoint_path=Path(args.checkpoint),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        trans_bound_mm=args.trans_bound_mm,
        rot_bound_deg=args.rot_bound_deg,
        dry_run=args.dry_run,
        output_plot=Path(args.output_plot),
        save_checkpoint=Path(args.save_checkpoint) if args.save_checkpoint is not None else None,
        device_name=args.device,
        branches=args.branches,
        eval_rot_bound_deg=args.eval_rot_bound_deg,
        seed=args.seed,
        quiet=args.quiet,
        mirror_dirs=tuple(args.mirror_dir),
        allow_synthetic=args.allow_synthetic,
        variant=args.variant,
        bn_reestimate=args.bn_reestimate,
    )


if __name__ == "__main__":
    main()
