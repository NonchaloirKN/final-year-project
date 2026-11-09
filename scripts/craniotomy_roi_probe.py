"""
Craniotomy-centred region-of-interest probe with real intra-operative ultrasound (Appendix B.4, part 2).

The whole-head frame of the primary pipeline places the slice centre at the centre of the MRI volume, whereas the
RESECT pre-resection ultrasound of the withheld patients covers only the tumour region beneath the craniotomy, so
whole-head planes cut almost none of it (scripts/real_ius_and_fov_probe.py). Here every plane is defined in a
craniotomy-centred frame, as a navigation system would provide it:

- Alignment. The pre-resection 3D iUS is aligned to the T2-FLAIR volume by a least-squares rigid fit to the expert
  MRI-iUS landmark pairs of RESECT and resampled onto the FLAIR grid.
- Frame. The centre c_US is the centroid of the active ultrasound signal (the tracker prior). The canonical
  227 x 227 plane spans 2 h_roi = 80 mm (0.35 mm pixels), the acoustic depth of the simulator. Poses are c_US plus a
  local perturbation of +/-10 mm and +/-20 deg; labels are absolute (volume-centred) positions in units of h_roi.
- Inputs. For each plane: the FLAIR reslice and its simulated frame over the full 80 mm square, and the FLAIR,
  simulated and real ultrasound frames restricted to the footprint of the real ultrasound in that plane (the same
  mask for all three, so no branch can use the mask outline).
- Networks. (R) a craniotomy-scale Stage 1 network per fold, trained exactly like the primary network (100 epochs,
  ImageNet initialisation, joint loss, 50-epoch rotation curriculum, seed 0) on 80 mm planes centred on random
  points at least 20 mm inside the head of the 22 training patients, with the 16 slices of a batch sharing a centre;
  (W) the whole-head Stage 1 network of the same fold, for reference.
- States, each from Stage 1 with the S8e calibration settings: zero-US calibration on simulated frames (full square
  or footprint); the footprint-calibrated model on real ultrasound, directly and after re-estimating NN2's
  batch-normalisation statistics on 160 real frames (no weight change); a readout adaptation (projection heads and
  NN3) on disjoint real frames; and the same with NN2's last residual stage (layer4) also trainable.

Version 2 (the final design, --version v2) changes three things: labels are the local offset (dt, R) about the
patch centre, with the translation residual scaled so that the 10 mm bound costs one unit (in Stage 1 the centres
are random, so this term only regularises; in Stage 2 the centre is fixed and the offset becomes learnable); the
simulator is time-gain compensated (no net Beer-Lambert attenuation); and the zero-US calibration also adapts
layer4 of both encoders, still on FLAIR and simulated frames only. v2 evaluates the zero-US model on simulated
footprint frames, the same zero-shot weights on real ultrasound (with and without BN re-estimation), and few-shot
adaptations on disjoint real frames.

Usage:
  python scripts/craniotomy_roi_probe.py --stage train [--version v2] --cases Case23 Case1 Case12 Case2
  python scripts/craniotomy_roi_probe.py --stage evaluate [--version v2] --cases Case23 Case1 Case12 Case2
Writes the 'craniotomy_roi' (v1) or 'craniotomy_roi_v2' key of docs/case23_benchmark.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_patient import freeze_encoders, per_slice_errors, reestimate_batchnorm, selection_loss
from scripts.case23_benchmark import BENCHMARK_JSON, CAL_CKPT, UNCAL_CKPT, cosine_diagnostics, load_benchmark, load_pipeline
from scripts.real_ius_and_fov_probe import per_frame_max_norm, read_tags, rigid_fit
from scripts.train_contrastive import get_loso_splits
from scripts.train_pose_pipeline import find_case_volume, load_or_create_volume, sample_random_6d_poses
from src.losses.pose_loss import MultiModalPoseLoss
from src.models.encoders import NN1, NN2
from src.models.pose_regressor import NN3
from src.simulator.sim_us import AcousticSimulator
from src.utils.dataset import load_canonical_nifti
from src.utils.metrics import volume_half_extent_mm
from src.utils.reslicer import VolumeReslicer

NIFTI = Path("data/resect_base/RESECT/NIFTI")
CKPT_DIR = Path("checkpoints/craniotomy_roi")
WHOLE_HEAD = {"Case23": UNCAL_CKPT, "Case1": Path("checkpoints/loso_folds/case1/best_model_pose_pipeline.pth"),
              "Case12": Path("checkpoints/loso_folds/case12/best_model_pose_pipeline.pth"),
              "Case2": Path("checkpoints/loso_folds/case2/best_model_pose_pipeline.pth")}
ROI_MM, LOCAL_T, LOCAL_R = 40.0, 10.0, 20.0
UNIT = (ROI_MM, ROI_MM, ROI_MM)
ACTIVE = 0.1
INSET_MM = 20.0
# v1 (bb43f7f): absolute (volume-centred) labels, the attenuated simulator, frozen backbones in calibration.
# v2: labels are local offsets about the patch centre (translation scaled so 10 mm = 1), the simulator is
#     time-gain compensated (no net attenuation), and the zero-US calibration may train layer4 of both encoders.
CFG = {"version": "v1", "local_labels": False, "tgc": False, "trans_ref_mm": 20.0, "ckpt_dir": CKPT_DIR,
       "key": "craniotomy_roi", "beta_rot": 1.0, "fixed_centres": False, "cal_epochs": 10}
DISC_MM = 15.0          # static disc (v3 control): covered by real iUS in 98-100% of planes, so its outline is pose-free


def set_version(version: str):
    if version == "v3":
        # v3: Stage 1 about each training patient's fixed craniotomy centre (data/derived/craniotomy_centres.json),
        #     so the local offset is learnable; rotation weighted by beta_rot = 6 (the identity guess costs 0.081 in
        #     rotation against 0.389 in translation at +/-20 deg); zero-US calibration for 20 epochs
        CFG.update({"version": "v3", "local_labels": True, "tgc": True, "trans_ref_mm": LOCAL_T, "beta_rot": 6.0,
                    "fixed_centres": True, "cal_epochs": 20, "ckpt_dir": Path("checkpoints/craniotomy_roi_v3"),
                    "key": "craniotomy_roi_v3"})
    if version == "v2":
        CFG.update({"version": "v2", "local_labels": True, "tgc": True, "trans_ref_mm": LOCAL_T,
                    "ckpt_dir": Path("checkpoints/craniotomy_roi_v2"), "key": "craniotomy_roi_v2"})


def roi_reslicer(device):
    return VolumeReslicer(slice_size=(227, 227), rotation_mode="rotation_6d", ref_mm=ROI_MM).to(device)


def base_config():
    return copy.deepcopy(torch.load(CAL_CKPT, map_location="cpu", weights_only=False)["config"])


def roi_simulator(device):
    """The primary simulator; with time-gain compensation (v2) the Beer-Lambert attenuation is fully compensated."""
    s = base_config()["simulator"]
    return AcousticSimulator(device=device, reflection_weight=s["reflection_weight"], fan_angle_deg=s["fan_angle_deg"],
                             apex_offset_frac=s["apex_offset_frac"], per_slice_norm=s["per_slice_norm"], f0=s["f0"],
                             alpha_wm=0.0 if CFG["tgc"] else s["alpha_wm"], sigma_rayleigh=s["sigma_rayleigh"])


def trans_scale(device):
    return torch.tensor(UNIT, device=device) / CFG["trans_ref_mm"]


# ---------------------------------------------------------------------------------------------- patient data
def align_case(case: str) -> dict:
    """Landmark-rigid alignment of the pre-resection iUS onto the FLAIR grid (cached in data/derived)."""
    fl_img = nib.load(str(NIFTI / case / "MRI" / f"{case}-FLAIR.nii.gz"))
    us_img = nib.load(str(NIFTI / case / "US" / f"{case}-US-before.nii.gz"))
    m_pts, u_pts = read_tags(NIFTI / case / "Landmarks" / f"{case}-MRI-beforeUS.tag")
    r, t = rigid_fit(u_pts, m_pts)
    loo = []
    for i in range(len(m_pts)):
        keep = np.arange(len(m_pts)) != i
        ri, ti = rigid_fit(u_pts[keep], m_pts[keep])
        loo.append(np.linalg.norm(m_pts[i] - (ri @ u_pts[i] + ti)))
    out_path = Path(f"data/derived/{case}-US-before_on_FLAIR.nii.gz")
    if not out_path.exists():
        rigid_inv = np.eye(4)
        rigid_inv[:3, :3], rigid_inv[:3, 3] = r.T, -r.T @ t
        m = np.linalg.inv(us_img.affine) @ rigid_inv @ fl_img.affine
        vol = ndimage.affine_transform(us_img.get_fdata(dtype=np.float32), m[:3, :3], m[:3, 3],
                                       output_shape=fl_img.shape, order=1, cval=0.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(vol.astype(np.float32), fl_img.affine, fl_img.header), str(out_path))
    return {"path": str(out_path), "landmarks": int(len(m_pts)),
            "header_error_mm": float(np.linalg.norm(m_pts - u_pts, axis=1).mean()),
            "fit_residual_mm": float(np.linalg.norm(m_pts - (u_pts @ r.T + t), axis=1).mean()),
            "leave_one_out_mm": float(np.mean(loo))}


class Patient:
    """A withheld patient in the craniotomy-centred frame: FLAIR, aligned iUS, centre c_US and plane sampler."""

    def __init__(self, case: str, device):
        self.case, self.device = case, device
        self.align = align_case(case)
        self.simulator = roi_simulator(device)
        self.reslicer = roi_reslicer(device)
        self.vol, self.shape, self.spacing = load_or_create_volume(NIFTI / case / "MRI" / f"{case}-FLAIR.nii.gz",
                                                                   device, canonical=True, plane="axial")
        us, zooms, _ = load_canonical_nifti(self.align["path"], plane="axial")
        assert us.shape == tuple(self.shape), "the aligned iUS must share the FLAIR grid"
        us = np.clip(us / np.percentile(us[us > 0], 99), 0.0, 1.0)
        self.us = torch.from_numpy(us).unsqueeze(0).unsqueeze(0).to(device)
        self.extent = volume_half_extent_mm(self.shape, self.spacing)
        d, h, w = np.nonzero(us > ACTIVE)
        dd, hh, ww = self.shape
        sz, sy, sx = zooms
        c = np.array([(w.mean() - (ww - 1) / 2) * sx, (h.mean() - (hh - 1) / 2) * sy, (d.mean() - (dd - 1) / 2) * sz])
        self.c_mm = c
        self.c = torch.tensor(c / ROI_MM, dtype=torch.float32, device=device)
        g = torch.arange(227, device=device, dtype=torch.float32) - 113.0
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        self.disc = (((xx ** 2 + yy ** 2).sqrt() * (2 * ROI_MM / 226)) <= DISC_MM).float()[None, None]

    def frames(self, p: torch.Tensor) -> dict:
        with torch.no_grad():
            mri = torch.cat([self.reslicer(self.vol, p[i:i + 40], half_extent_mm=self.extent) for i in range(0, len(p), 40)])
            real = torch.cat([self.reslicer(self.us, p[i:i + 40], half_extent_mm=self.extent) for i in range(0, len(p), 40)])
            sim = torch.cat([self.simulator(mri[i:i + 16]) for i in range(0, len(p), 16)])
        m = (real > 0).float()
        d = self.disc
        return {"p_gt": p, "mri": mri, "sim": sim, "mri_fp": mri * m, "sim_fp": per_frame_max_norm(sim * m),
                "real": per_frame_max_norm(real), "fill": (real > ACTIVE).flatten(1).float().mean(1),
                "mri_disc": mri * d, "sim_disc": per_frame_max_norm(sim * d), "real_disc": per_frame_max_norm(real * d),
                "disc_cover": ((real > 0).float() * d).flatten(1).sum(1) / d.sum()}

    def draw(self, n: int, seed=None, sim_seed=None) -> dict:
        """n poses about c_US (seeded or from the current stream); speckle from sim_seed if given."""
        if seed is not None:
            torch.manual_seed(seed)
        p = sample_random_6d_poses(n, self.shape, self.spacing, LOCAL_T, LOCAL_R, device=self.device, ref_mm=ROI_MM)
        ps = p.clone()
        ps[:, :3] += self.c                                  # slice pose: centroid + local offset
        if sim_seed is not None:
            torch.manual_seed(sim_seed)
        out = self.frames(ps)
        if CFG["local_labels"]:
            out["p_gt"] = p                                  # label: the local offset about the craniotomy centre
        return out


# ---------------------------------------------------------------------------------------------- Stage 1 (R)
def head_centres(vol: torch.Tensor, spacing) -> torch.Tensor:
    """Candidate ROI centres (mm, volume-centred, x/y/z = W/H/D) at least INSET_MM inside the head."""
    v = vol[0, 0].cpu().numpy()
    head = ndimage.binary_fill_holes(ndimage.binary_closing(v > 0.1, iterations=3))
    lab, n = ndimage.label(head)
    head = lab == (1 + int(np.argmax(ndimage.sum(head, lab, range(1, n + 1)))))
    inside = ndimage.distance_transform_edt(head, sampling=spacing) > INSET_MM
    d, h, w = np.nonzero(inside)
    sel = np.arange(0, len(d), 7)
    dd, hh, ww = v.shape
    pts = np.stack([(w[sel] - (ww - 1) / 2) * spacing[2], (h[sel] - (hh - 1) / 2) * spacing[1],
                    (d[sel] - (dd - 1) / 2) * spacing[0]], 1)
    return torch.tensor(pts, dtype=torch.float32)


def train_roi_stage1(withheld: str, device, epochs=100, steps=25, batch=16, curriculum=50, seed=0) -> Path:
    cfg = base_config()
    simulator = roi_simulator(device)
    reslicer = roi_reslicer(device)
    train_cases, _ = get_loso_splits(NIFTI, val_case=withheld)
    data = {}
    for c in train_cases:
        vol, shape, spacing = load_or_create_volume(find_case_volume(c, NIFTI), device, canonical=True, plane="axial")
        data[c] = (vol, shape, spacing, volume_half_extent_mm(shape, spacing), head_centres(vol, spacing).to(device))
    torch.manual_seed(seed)
    np.random.seed(seed)
    nn1, nn2 = NN1(embedding_dim=128, pretrained=True).to(device), NN2(embedding_dim=128, pretrained=True).to(device)
    nn3 = NN3(in_features=128, symmetric_trunk=True).to(device)
    params = list(nn1.parameters()) + list(nn2.parameters()) + list(nn3.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = MultiModalPoseLoss(alpha=1.0, beta_rot=CFG["beta_rot"], contrastive_weight=0.1, temperature=0.1,
                              axis_weights=(1.0, 1.0, 1.5))
    fixed = {}
    if CFG["fixed_centres"]:
        cj = json.loads(Path("data/derived/craniotomy_centres.json").read_text(encoding="utf-8"))
        fixed = {c: torch.tensor(cj[c]["centre_mm"], dtype=torch.float32, device=device)[None] for c in train_cases}
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ts = trans_scale(device)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        rot = LOCAL_R * min(1.0, epoch / curriculum)
        order = [train_cases[i] for i in torch.randperm(len(train_cases)).tolist()]
        tot = 0.0
        for step in range(steps):
            case = order[step % len(order)]
            vol, shape, spacing, ext, cen = data[case]
            centre = (fixed[case] if fixed else cen[torch.randint(len(cen), (1,), device=device)]) / ROI_MM
            p = sample_random_6d_poses(batch, shape, spacing, LOCAL_T, rot, device=device, ref_mm=ROI_MM)
            ps = p.clone()
            ps[:, :3] += centre
            if not CFG["local_labels"]:
                p = ps
            mri = reslicer(vol, ps, half_extent_mm=ext)
            us = simulator(mri)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                e1, e2 = nn1(mri), nn2(us)
                p_us, p_mri = nn3(e2), nn3(e1)
            loss, det = crit(emb1=e1.float(), emb2=e2.float(), p_pred=p_us.float(), p_gt=p, p_pred_mri=p_mri.float(),
                             return_details=True, trans_scale=ts)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tot += det["total_loss"]
        sched.step()
        hist.append(tot / steps)
        if epoch >= curriculum and hist[-1] < best:
            best = hist[-1]
            best_state = {"epoch": epoch, "nn1_state_dict": copy.deepcopy(nn1.state_dict()),
                          "nn2_state_dict": copy.deepcopy(nn2.state_dict()),
                          "nn3_state_dict": copy.deepcopy(nn3.state_dict())}
        if epoch % 10 == 0:
            print(f"    [{withheld}] epoch {epoch:3d} loss {hist[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    cfg.update({"ref_mm": ROI_MM, "craniotomy_roi": {"roi_mm": 2 * ROI_MM, "local_t_mm": LOCAL_T, "local_rot_deg": LOCAL_R,
                                                      "inset_mm": INSET_MM, "withheld": withheld, "seconds": time.time() - t0,
                                                      "version": CFG["version"], "local_labels": CFG["local_labels"],
                                                      "tgc": CFG["tgc"], "beta_rot": CFG["beta_rot"],
                                                      "fixed_centres": CFG["fixed_centres"]}})
    out = CFG["ckpt_dir"] / withheld.lower() / "best_model_pose_pipeline.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**best_state, "loss": best, "history": {"total_loss": hist}, "config": cfg}, out)
    print(f"[*] {withheld}: craniotomy-scale Stage 1 saved (epoch {best_state['epoch']}, {time.time() - t0:.0f} s)")
    return out


# ---------------------------------------------------------------------------------------------- Stage 2 / probes
def calibrate(init_ckpt: Path, P: Patient, keys, bn: dict, sel: dict, variants=("nn3", "proj_nn3"), layer4=False,
              epochs=10, steps=25, seed=0, unfreeze=()):
    """Stage 2 from a Stage 1 checkpoint on the given (mri, us) input keys; returns (nn1, nn2, nn3), info."""
    dev = P.device
    (n1, n2, n3), ck = load_pipeline(init_ckpt, dev)
    cfg = ck.get("config") or {}
    aw = (1.0, 1.0, float(cfg.get("tz_weight", 1.5)))
    ts = trans_scale(dev)
    pose = MultiModalPoseLoss(alpha=1.0, beta_rot=CFG["beta_rot"], trans_scale=ts, axis_weights=aw)
    full = MultiModalPoseLoss(alpha=1.0, beta_rot=CFG["beta_rot"], trans_scale=ts, axis_weights=aw, contrastive_weight=0.1)
    km, ku = keys
    reestimate_batchnorm(n1, bn[km])
    reestimate_batchnorm(n2, bn[ku])
    freeze_encoders(n1, n2)
    sel_set = {"p_gt": sel["p_gt"], "mri": sel[km], "us": sel[ku]}
    init = {k: copy.deepcopy(n.state_dict()) for k, n in (("n1", n1), ("n2", n2), ("n3", n3))}
    runs, t0 = {}, time.time()
    for variant in variants:
        for k, n in (("n1", n1), ("n2", n2), ("n3", n3)):
            n.load_state_dict(init[k])
        proj = variant == "proj_nn3"
        for p in n1.parameters():
            p.requires_grad = False
        for p in n2.parameters():
            p.requires_grad = False
        trainable = list(n3.parameters())
        if proj:
            trainable += list(n1.projector.parameters()) + list(n2.projector.parameters())
        blocks = set(unfreeze) | ({("n2", "layer4")} if layer4 else set())
        for net, layer in sorted(blocks):
            trainable += list(getattr({"n1": n1, "n2": n2}[net].backbone, layer).parameters())
        for p in trainable:
            p.requires_grad = True
        torch.manual_seed(seed)
        opt = AdamW(trainable, lr=3e-4, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        best_sel = selection_loss(pose, sel_set, "both", n1, n2, n3, ts)
        best = {k: copy.deepcopy(n.state_dict()) for k, n in (("n1", n1), ("n2", n2), ("n3", n3))}
        for _ in range(epochs):
            n3.train()
            for _ in range(steps):
                s = P.draw(16)
                with torch.set_grad_enabled(any(b[0] == "n1" for b in blocks)):
                    f1 = n1.backbone(s[km])
                f2 = n2.backbone(s[ku]) if any(b[0] == "n2" for b in blocks) else n2.backbone(s[ku]).detach()
                e1, e2 = F.normalize(n1.projector(f1), dim=-1), F.normalize(n2.projector(f2), dim=-1)
                crit = full if proj else pose
                loss = crit(emb1=e1 if proj else None, emb2=e2 if proj else None, p_pred=n3(e2), p_gt=s["p_gt"],
                            p_pred_mri=n3(e1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                opt.step()
            sch.step()
            n1.eval(), n2.eval()
            v = selection_loss(pose, sel_set, "both", n1, n2, n3, ts)
            if v < best_sel:
                best_sel, best = v, {k: copy.deepcopy(n.state_dict()) for k, n in (("n1", n1), ("n2", n2), ("n3", n3))}
        runs[variant] = (best_sel, best)
    chosen = min(runs, key=lambda k: runs[k][0])
    for k, n in (("n1", n1), ("n2", n2), ("n3", n3)):
        n.load_state_dict(runs[chosen][1][k])
        n.eval()
    return (n1, n2, n3), {"variant": chosen, "selection_loss": {k: v[0] for k, v in runs.items()},
                          "unfrozen": sorted(f"{a}.{b}" for a, b in blocks), "seconds": time.time() - t0}


def measure(nets, s: dict, km: str, ku: str) -> dict:
    n1, n2, n3 = nets
    with torch.no_grad():
        z1, z2 = n1(s[km]), n2(s[ku])
        out = {}
        for branch, z in (("mri", z1), ("us", z2)):
            e = per_slice_errors(n3(z), s["p_gt"], UNIT)
            out[branch] = {k: float(v.mean()) for k, v in e.items()}
            out[branch]["median_mte"] = float(np.median(e["mte"]))
        out["latent"] = cosine_diagnostics(z1, z2)
    return out


def trivial(P: Patient, s: dict) -> dict:
    guess = s["p_gt"].clone()
    guess[:, :3] = 0.0 if CFG["local_labels"] else P.c
    guess[:, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=guess.dtype, device=guess.device)
    e = per_slice_errors(guess, s["p_gt"], UNIT)
    return {"mte": float(e["mte"].mean()), "geo": float(e["geo"].mean())}


def evaluate_case(case: str, device, with_whole_head: bool) -> dict:
    P = Patient(case, device)
    ev, bn, sel = P.draw(40, seed=42, sim_seed=1234), P.draw(160, seed=11, sim_seed=5678), P.draw(160, seed=7, sim_seed=4321)
    out = {"alignment": P.align, "c_us_mm": [float(x) for x in P.c_mm],
           "fill_median": float(ev["fill"].median()), "fill_min": float(ev["fill"].min()), "trivial": trivial(P, ev)}
    backbones = {"craniotomy_stage1": CFG["ckpt_dir"] / case.lower() / "best_model_pose_pipeline.pth"}
    if with_whole_head:
        backbones["whole_head_stage1"] = WHOLE_HEAD[case]
    for name, ck in backbones.items():
        r = {}
        nets, info = calibrate(ck, P, ("mri", "sim"), bn, sel)
        r["sim_full_square"] = {**measure(nets, ev, "mri", "sim"), "calibration": info}
        nets, info = calibrate(ck, P, ("mri_fp", "sim_fp"), bn, sel)
        r["sim_footprint"] = {**measure(nets, ev, "mri_fp", "sim_fp"), "calibration": info}
        r["real_zero_us"] = measure(nets, ev, "mri_fp", "real")
        n2 = copy.deepcopy(nets[1])
        reestimate_batchnorm(n2, bn["real"])
        r["real_zero_us_bn"] = measure((nets[0], n2, nets[2]), ev, "mri_fp", "real")
        nets, info = calibrate(ck, P, ("mri_fp", "real"), bn, sel, variants=("proj_nn3",))
        r["real_readout_adapted"] = {**measure(nets, ev, "mri_fp", "real"), "calibration": info}
        if name == "craniotomy_stage1":
            nets, info = calibrate(ck, P, ("mri_fp", "real"), bn, sel, variants=("proj_nn3",), layer4=True)
            r["real_readout_layer4"] = {**measure(nets, ev, "mri_fp", "real"), "calibration": info}
        out[name] = r
        for k, v in r.items():
            print(f"  {case} {name:<18} {k:<22} MRI {v['mri']['mte']:6.2f} mm | US {v['us']['mte']:6.2f} mm "
                  f"{v['us']['geo']:5.2f} deg | top-1 {v['latent']['top1_retrieval_pct']:5.1f}%", flush=True)
    if case == "Case23":
        i = int(ev["fill"].argmax())
        torch.save({k: ev[k][i:i + 1].cpu() for k in ("mri", "sim", "real", "p_gt")} | {"fill": float(ev["fill"][i])},
                   "data/derived/case23_craniotomy_example.pt")
    print(f"[*] {case}: c_US {np.round(P.c_mm, 1)} mm, fill median {out['fill_median']:.2f}, trivial {out['trivial']}")
    return out


def evaluate_case_v2(case: str, device) -> dict:
    """v2: local labels, time-gain-compensated simulator, zero-US calibration with and without layer4 of both encoders."""
    P = Patient(case, device)
    ev, bn, sel = P.draw(40, seed=42, sim_seed=1234), P.draw(160, seed=11, sim_seed=5678), P.draw(160, seed=7, sim_seed=4321)
    fp = (ev["real"] > 0).float()
    bright = {k: float((ev[k] * fp).sum() / fp.sum()) for k in ("sim_fp", "real")}
    out = {"alignment": P.align, "c_us_mm": [float(x) for x in P.c_mm], "fill_median": float(ev["fill"].median()),
           "trivial": trivial(P, ev), "footprint_brightness": bright}
    ck = CFG["ckpt_dir"] / case.lower() / "best_model_pose_pipeline.pth"
    l4 = (("n1", "layer4"), ("n2", "layer4"))
    r = {}
    nets, info = calibrate(ck, P, ("mri_fp", "sim_fp"), bn, sel, variants=("proj_nn3",))
    r["zero_us_frozen_sim"] = {**measure(nets, ev, "mri_fp", "sim_fp"), "calibration": info}
    n2 = copy.deepcopy(nets[1])
    reestimate_batchnorm(n2, bn["real"])
    r["zero_us_frozen_real_bn"] = measure((nets[0], n2, nets[2]), ev, "mri_fp", "real")
    nets, info = calibrate(ck, P, ("mri", "sim"), bn, sel, variants=("proj_nn3",), unfreeze=l4)
    r["zero_us_layer4_sim_full_square"] = {**measure(nets, ev, "mri", "sim"), "calibration": info}
    nets, info = calibrate(ck, P, ("mri_fp", "sim_fp"), bn, sel, variants=("proj_nn3",), unfreeze=l4)
    r["zero_us_layer4_sim"] = {**measure(nets, ev, "mri_fp", "sim_fp"), "calibration": info}
    r["zero_us_layer4_real"] = measure(nets, ev, "mri_fp", "real")
    n2 = copy.deepcopy(nets[1])
    reestimate_batchnorm(n2, bn["real"])
    r["zero_us_layer4_real_bn"] = measure((nets[0], n2, nets[2]), ev, "mri_fp", "real")
    nets, info = calibrate(ck, P, ("mri_fp", "real"), bn, sel, variants=("proj_nn3",))
    r["real_readout_adapted"] = {**measure(nets, ev, "mri_fp", "real"), "calibration": info}
    nets, info = calibrate(ck, P, ("mri_fp", "real"), bn, sel, variants=("proj_nn3",), layer4=True)
    r["real_readout_layer4"] = {**measure(nets, ev, "mri_fp", "real"), "calibration": info}
    out["craniotomy_stage1"] = r
    for k, v in r.items():
        print(f"  {case} v2 {k:<32} MRI {v['mri']['mte']:5.2f} mm {v['mri']['geo']:5.2f} deg | US {v['us']['mte']:5.2f} mm "
              f"{v['us']['geo']:5.2f} deg | top-1 {v['latent']['top1_retrieval_pct']:5.1f}%", flush=True)
    if case == "Case23":
        i = int(ev["fill"].argmax())
        torch.save({k: ev[k][i:i + 1].cpu() for k in ("mri", "sim", "real", "p_gt")} | {"fill": float(ev["fill"][i])},
                   "data/derived/case23_craniotomy_example_v2.pt")
    print(f"[*] {case}: trivial {out['trivial']}, footprint brightness {bright}")
    return out


def evaluate_case_v3(case: str, device) -> dict:
    """v3: fixed Stage 1 centres, beta_rot = 6, 20-epoch zero-US calibration with layer4; footprint and static-disc inputs."""
    P = Patient(case, device)
    ev, bn, sel = P.draw(40, seed=42, sim_seed=1234), P.draw(160, seed=11, sim_seed=5678), P.draw(160, seed=7, sim_seed=4321)
    out = {"alignment": P.align, "c_us_mm": [float(x) for x in P.c_mm], "fill_median": float(ev["fill"].median()),
           "disc_cover_mean": float(ev["disc_cover"].mean()), "disc_cover_min": float(ev["disc_cover"].min()),
           "trivial": trivial(P, ev)}
    ck = CFG["ckpt_dir"] / case.lower() / "best_model_pose_pipeline.pth"
    l4 = (("n1", "layer4"), ("n2", "layer4"))
    r = {}
    for tag, km, ks, kr in (("footprint", "mri_fp", "sim_fp", "real"), ("disc", "mri_disc", "sim_disc", "real_disc")):
        nets, info = calibrate(ck, P, (km, ks), bn, sel, variants=("proj_nn3",), unfreeze=l4, epochs=CFG["cal_epochs"])
        r[f"zero_us_{tag}_sim"] = {**measure(nets, ev, km, ks), "calibration": info}
        r[f"zero_us_{tag}_real"] = measure(nets, ev, km, kr)
        n2 = copy.deepcopy(nets[1])
        reestimate_batchnorm(n2, bn[kr])
        r[f"zero_us_{tag}_real_bn"] = measure((nets[0], n2, nets[2]), ev, km, kr)
        nets, info = calibrate(ck, P, (km, kr), bn, sel, variants=("proj_nn3",), layer4=True)
        r[f"few_shot_{tag}_real"] = {**measure(nets, ev, km, kr), "calibration": info}
    out["craniotomy_stage1"] = r
    for k, v in r.items():
        u = v["us"]
        print(f"  {case} v3 {k:<26} US {u['mte']:5.2f} mm {u['geo']:5.2f} deg (p {u['pitch']:.2f} r {u['roll']:.2f} "
              f"y {u['yaw']:.2f}) | MRI {v['mri']['mte']:5.2f} mm {v['mri']['geo']:5.2f} deg | "
              f"top-1 {v['latent']['top1_retrieval_pct']:5.1f}%", flush=True)
    if case == "Case23":
        i = int(ev["fill"].argmax())
        torch.save({k: ev[k][i:i + 1].cpu() for k in ("mri", "sim", "real", "p_gt")} | {"fill": float(ev["fill"][i])},
                   "data/derived/case23_craniotomy_example_v3.pt")
    print(f"[*] {case}: trivial {out['trivial']}, disc cover {out['disc_cover_mean']:.3f} (min {out['disc_cover_min']:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=["train", "evaluate"], required=True)
    ap.add_argument("--cases", nargs="+", default=["Case23", "Case1", "Case12", "Case2"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--version", choices=["v1", "v2", "v3"], default="v1")
    args = ap.parse_args()
    set_version(args.version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "train":
        for c in args.cases:
            train_roi_stage1(c, device, epochs=args.epochs, curriculum=min(50, max(1, args.epochs // 2)))
        return
    if CFG["version"] == "v3":
        res = {c: evaluate_case_v3(c, device) for c in args.cases}
    elif CFG["version"] == "v2":
        res = {c: evaluate_case_v2(c, device) for c in args.cases}
    else:
        res = {c: evaluate_case(c, device, with_whole_head=(c == "Case23")) for c in args.cases}
    results = load_benchmark()
    prev = results.get(CFG["key"], {}).get("cases", {})
    prev.update(res)
    results[CFG["key"]] = {"roi_mm": 2 * ROI_MM, "local_t_mm": LOCAL_T, "local_rot_deg": LOCAL_R, "inset_mm": INSET_MM,
                           "local_labels": CFG["local_labels"], "tgc": CFG["tgc"], "trans_ref_mm": CFG["trans_ref_mm"],
                           "beta_rot": CFG["beta_rot"], "fixed_centres": CFG["fixed_centres"],
                           "cal_epochs": CFG["cal_epochs"], "disc_mm": DISC_MM,
                           "cases": prev}
    BENCHMARK_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[*] {CFG['key']} written to {BENCHMARK_JSON}")


if __name__ == "__main__":
    main()
