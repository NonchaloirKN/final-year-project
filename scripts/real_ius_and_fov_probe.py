"""
Field-of-view and sim-to-real probes on the withheld Case 23 (Appendix B.4).

Part A, field-of-view mask. The simulated frames show the whole 255 mm slice, including the skull outline, whereas a
real probe on the cortex sees an interior sector without it. Every MRI reslice and simulated frame of Protocol B is
restricted to a central disc of 80 mm diameter (the 8 cm acoustic depth of the simulator), which removes the skull
and the outer cortex, and the primary pipeline is evaluated (i) as calibrated on full slices, (ii) after
re-estimating the batch-normalisation statistics on 160 masked slices (no weight change) and (iii) after a full
zero-US calibration on masked slices, from the Stage 1 network with the S8e settings.

Part B, real intra-operative ultrasound. The RESECT pre-resection 3D iUS volume of Case 23 is aligned to the FLAIR
volume by a least-squares rigid fit to the 15 expert MRI-iUS landmark pairs (the tracked header alignment leaves a
landmark error of about 7 mm) and resampled onto the FLAIR grid. Protocol B poses are drawn from the usual seeded
stream (+/-20 mm, +/-30 deg) and a pose is kept only if its plane cuts at least 500 pixels of real iUS signal
(normalised intensity > 0.1). On identical poses three inputs reach NN2: simulated frames of the full slice,
simulated frames restricted to the footprint of the real iUS, and the real iUS reslices. This separates the field of
view from the appearance gap. Each is evaluated with the FLAIR-calibrated model, after batch-normalisation
re-estimation on 160 such frames, and after a calibration on them (for the real iUS this uses intra-operative
ultrasound of the same volume on disjoint poses, so it probes the frozen features rather than a clinical protocol).

Writes the 'fov_and_real_ius' key of docs/case23_benchmark.json.
"""

from __future__ import annotations

import copy
import json
import math
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

from scripts.calibrate_patient import (build_eval_set, freeze_encoders, per_slice_errors, pipeline_from_config,
                                       reestimate_batchnorm, selection_loss, volume_options)
from scripts.case23_benchmark import (BENCHMARK_JSON, CAL_CKPT, UNCAL_CKPT, cosine_diagnostics, load_benchmark,
                                      load_pipeline)
from scripts.cross_sequence_t1 import nmi
from scripts.train_pose_pipeline import load_or_create_volume, sample_random_6d_poses
from src.losses.pose_loss import MultiModalPoseLoss
from src.utils.dataset import load_canonical_nifti
from src.utils.metrics import pose_unit_mm, volume_half_extent_mm

CASE_DIR = Path("data/resect_base/RESECT/NIFTI/Case23")
FLAIR = CASE_DIR / "MRI" / "Case23-FLAIR.nii.gz"
US_BEFORE = CASE_DIR / "US" / "Case23-US-before.nii.gz"
TAGS = CASE_DIR / "Landmarks" / "Case23-MRI-beforeUS.tag"
DERIVED = Path("data/derived/Case23-US-before_on_FLAIR.nii.gz")
FOV_DIAMETER_MM = 80.0
MIN_ACTIVE_PX, ACTIVE_LEVEL = 500, 0.1
TRANS_MM, ROT_DEG = 20.0, 30.0


# ---------------------------------------------------------------------------------------------- alignment
def read_tags(path: Path):
    """MNI tag file -> (MRI points, iUS points), world coordinates in mm."""
    rows = [r.split() for r in path.read_text().split("Points =")[1].strip().splitlines()]
    pts = np.array([[float(x) for x in r[:6]] for r in rows if len(r) >= 6])
    return pts[:, :3], pts[:, 3:6]


def rigid_fit(src: np.ndarray, dst: np.ndarray):
    """Least-squares rigid transform (Kabsch) with dst ~ R src + t."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cd - r @ cs


def align_ius() -> dict:
    """Resample the pre-resection iUS onto the FLAIR grid through the landmark rigid fit; report alignment quality."""
    fl_img, us_img = nib.load(str(FLAIR)), nib.load(str(US_BEFORE))
    m_pts, u_pts = read_tags(TAGS)
    r, t = rigid_fit(u_pts, m_pts)
    header_err = np.linalg.norm(m_pts - u_pts, axis=1)
    fit_err = np.linalg.norm(m_pts - (u_pts @ r.T + t), axis=1)
    loo = []
    for i in range(len(m_pts)):
        keep = np.arange(len(m_pts)) != i
        ri, ti = rigid_fit(u_pts[keep], m_pts[keep])
        loo.append(np.linalg.norm(m_pts[i] - (ri @ u_pts[i] + ti)))
    # FLAIR voxel -> world (MRI) -> world (iUS) through the inverse fit -> iUS voxel
    rigid_inv = np.eye(4)
    rigid_inv[:3, :3], rigid_inv[:3, 3] = r.T, -r.T @ t
    us = us_img.get_fdata(dtype=np.float32)
    fl = fl_img.get_fdata(dtype=np.float32)
    out = {}
    for key, world in (("landmark_rigid", rigid_inv), ("header", np.eye(4))):
        m = np.linalg.inv(us_img.affine) @ world @ fl_img.affine
        vol = ndimage.affine_transform(us, m[:3, :3], m[:3, 3], output_shape=fl.shape, order=1, cval=0.0)
        out[key] = vol
    mask = out["landmark_rigid"] > 0
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out["landmark_rigid"].astype(np.float32), fl_img.affine, fl_img.header), str(DERIVED))
    angle = math.degrees(math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    return {"landmarks": int(len(m_pts)), "ius_shape": list(us.shape),
            "ius_spacing_mm": [float(z) for z in us_img.header.get_zooms()[:3]],
            "header_landmark_error_mm": {"mean": float(header_err.mean()), "sd": float(header_err.std()),
                                         "max": float(header_err.max())},
            "rigid_fit_rotation_deg": angle,
            "rigid_fit_mean_landmark_displacement_mm": float(np.linalg.norm(u_pts @ r.T + t - u_pts, axis=1).mean()),
            "fit_residual_mm": {"mean": float(fit_err.mean()), "max": float(fit_err.max())},
            "leave_one_out_error_mm": {"mean": float(np.mean(loo)), "max": float(np.max(loo))},
            "nmi_with_flair": {k: nmi(fl, out[k], mask & (out[k] > 0)) for k in out},
            "superior_offset_of_landmarks_mm": {
                "mean": float(np.mean(m_pts[:, 2] - (fl_img.affine @ np.r_[(np.array(fl.shape) - 1) / 2, 1])[2])),
                "min": float(np.min(m_pts[:, 2] - (fl_img.affine @ np.r_[(np.array(fl.shape) - 1) / 2, 1])[2])),
                "max": float(np.max(m_pts[:, 2] - (fl_img.affine @ np.r_[(np.array(fl.shape) - 1) / 2, 1])[2]))}}


def load_ius_volume(plane: str, device) -> torch.Tensor:
    """The aligned iUS in the canonical axial frame, scaled by the 99th percentile of its non-zero voxels."""
    data, _, _ = load_canonical_nifti(DERIVED, plane=plane)
    flair, _, _ = load_canonical_nifti(FLAIR, plane=plane)
    assert data.shape == flair.shape, "the aligned iUS must share the FLAIR grid"
    scale = np.percentile(data[data > 0], 99)
    return torch.from_numpy(np.clip(data / scale, 0.0, 1.0)).unsqueeze(0).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------------------------- inputs
def fov_disc(n: int = 227, ref_mm: float = 127.5, diameter_mm: float = FOV_DIAMETER_MM, device=None) -> torch.Tensor:
    """Binary central disc on the canonical 227 x 227 grid (255 mm wide)."""
    px = 2 * ref_mm / (n - 1)
    c = torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return ((xx ** 2 + yy ** 2).sqrt() * px <= diameter_mm / 2).float()[None, None]


def per_frame_max_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1, 1)


class Context:
    def __init__(self, device):
        self.device = device
        (self.c1, self.c2, self.c3), ck = load_pipeline(CAL_CKPT, device)
        self.cfg = ck.get("config")
        (self.reslicer, self.simulator, self.ref_mm, self.symmetric, self.axis_weights,
         self.cw) = pipeline_from_config(self.cfg, device)
        self.vol, self.shape, self.spacing = load_or_create_volume(FLAIR, device, **volume_options(self.cfg))
        self.extent = volume_half_extent_mm(self.shape, self.spacing)
        self.unit = pose_unit_mm(self.shape, self.spacing, self.ref_mm)
        self.trans_scale = torch.tensor(self.unit, device=device) / TRANS_MM
        self.disc = fov_disc(ref_mm=self.ref_mm, device=device)
        self.ius = None

    def slices(self, vol, p):
        with torch.no_grad():
            return self.reslicer(vol, p, half_extent_mm=self.extent)


def sample_valid(ctx: Context, n: int, seed: int, batch: int = 4, sim_seed: int = 1234):
    """Seeded Protocol-B-style poses whose plane cuts >= MIN_ACTIVE_PX real-iUS pixels, with all three inputs."""
    torch.manual_seed(seed)
    poses, mri, real = [], [], []
    tried = 0
    while sum(len(p) for p in poses) < n:
        p = sample_random_6d_poses(batch, ctx.shape, ctx.spacing, TRANS_MM, ROT_DEG, device=ctx.device, ref_mm=ctx.ref_mm)
        tried += batch
        r = ctx.slices(ctx.ius, p)
        ok = (r > ACTIVE_LEVEL).flatten(1).sum(1) >= MIN_ACTIVE_PX
        if ok.any():
            poses.append(p[ok]); mri.append(ctx.slices(ctx.vol, p[ok])); real.append(r[ok])
    p, m, r = torch.cat(poses)[:n], torch.cat(mri)[:n], torch.cat(real)[:n]
    torch.manual_seed(sim_seed)
    with torch.no_grad():
        sim = torch.cat([ctx.simulator(m[i:i + 4]) for i in range(0, n, 4)])
    footprint = (r > 0).float()
    return {"p_gt": p, "mri": m, "sim_full": sim, "sim_footprint": per_frame_max_norm(sim * footprint),
            "real": per_frame_max_norm(r), "acceptance": n / tried,
            "active_px_median": float((r > ACTIVE_LEVEL).flatten(1).sum(1).float().median())}


# ---------------------------------------------------------------------------------------------- evaluation
def measure(ctx, n1, n2, n3, mri, us, p_gt):
    with torch.no_grad():
        z1, z2 = n1(mri), n2(us)
        out = {}
        for branch, z in (("mri", z1), ("us", z2)):
            e = per_slice_errors(n3(z), p_gt, ctx.unit)
            out[branch] = {k: float(v.mean()) for k, v in e.items()}
            out[branch].update({"median_mte": float(np.median(e["mte"])), "max_mte": float(e["mte"].max())})
        out["latent"] = cosine_diagnostics(z1, z2)
    return out


def trivial(ctx, p_gt):
    zero = p_gt.clone()
    zero[:, :3] = 0
    zero[:, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=p_gt.dtype, device=p_gt.device)
    e = per_slice_errors(zero, p_gt, ctx.unit)
    return {"mte": float(e["mte"].mean()), "geo": float(e["geo"].mean())}


def calibrate(ctx: Context, train_batch, bn_mri, bn_us, sel, epochs: int = 10, steps: int = 25, seed: int = 0):
    """Zero-US calibration from the Stage 1 network with the S8e settings, on arbitrary input frames.
    train_batch(16) -> (p_gt, mri, us); sel has 'p_gt', 'mri', 'us'. Returns the selected (nn1, nn2, nn3)."""
    (n1, n2, n3), _ = load_pipeline(UNCAL_CKPT, ctx.device)
    reestimate_batchnorm(n1, bn_mri)
    reestimate_batchnorm(n2, bn_us)
    freeze_encoders(n1, n2)
    pose_crit = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, trans_scale=ctx.trans_scale, axis_weights=ctx.axis_weights)
    full_crit = MultiModalPoseLoss(alpha=1.0, beta_rot=1.0, trans_scale=ctx.trans_scale, axis_weights=ctx.axis_weights,
                                   contrastive_weight=ctx.cw)
    init = {k: copy.deepcopy(n.state_dict()) for k, n in (("n1", n1), ("n2", n2), ("n3", n3))}
    runs = {}
    t0 = time.time()
    for variant in ("nn3", "proj_nn3"):
        for k, n in (("n1", n1), ("n2", n2), ("n3", n3)):
            n.load_state_dict(init[k])
        proj = variant == "proj_nn3"
        for p in list(n1.projector.parameters()) + list(n2.projector.parameters()):
            p.requires_grad = proj
        for p in n3.parameters():
            p.requires_grad = True
        params = list(n3.parameters()) + (list(n1.projector.parameters()) + list(n2.projector.parameters()) if proj else [])
        torch.manual_seed(seed)
        opt = AdamW(params, lr=3e-4, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        best_sel = selection_loss(pose_crit, sel, "both", n1, n2, n3, ctx.trans_scale)
        best = {k: copy.deepcopy(n.state_dict()) for k, n in (("n1", n1), ("n2", n2), ("n3", n3))}
        for _ in range(epochs):
            n3.train()
            for _ in range(steps):
                p_gt, mri, us = train_batch(16)
                with torch.no_grad():
                    f1, f2 = n1.backbone(mri), n2.backbone(us)
                e1, e2 = F.normalize(n1.projector(f1), dim=-1), F.normalize(n2.projector(f2), dim=-1)
                crit = full_crit if proj else pose_crit
                loss = crit(emb1=e1 if proj else None, emb2=e2 if proj else None, p_pred=n3(e2), p_gt=p_gt,
                            p_pred_mri=n3(e1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
            sched.step()
            s = selection_loss(pose_crit, sel, "both", n1, n2, n3, ctx.trans_scale)
            if s < best_sel:
                best_sel, best = s, {k: copy.deepcopy(n.state_dict()) for k, n in (("n1", n1), ("n2", n2), ("n3", n3))}
        runs[variant] = (best_sel, best)
    chosen = min(runs, key=lambda k: runs[k][0])
    for k, n in (("n1", n1), ("n2", n2), ("n3", n3)):
        n.load_state_dict(runs[chosen][1][k])
        n.eval()
    info = {"variant": chosen, "selection_loss": {k: v[0] for k, v in runs.items()}, "seconds": time.time() - t0}
    return (n1, n2, n3), info


# ---------------------------------------------------------------------------------------------- experiments
def part_a(ctx: Context) -> dict:
    d = ctx.disc
    b = build_eval_set(ctx.reslicer, ctx.vol, ctx.shape, ctx.spacing, 10, 4, TRANS_MM, ROT_DEG, seed=42,
                       simulator=ctx.simulator, sim_seed=1234, device=ctx.device)
    bn = build_eval_set(ctx.reslicer, ctx.vol, ctx.shape, ctx.spacing, 40, 4, TRANS_MM, ROT_DEG, seed=11,
                        simulator=ctx.simulator, sim_seed=5678, device=ctx.device)
    sel = build_eval_set(ctx.reslicer, ctx.vol, ctx.shape, ctx.spacing, 40, 4, TRANS_MM, ROT_DEG, seed=7,
                         simulator=ctx.simulator, sim_seed=4321, device=ctx.device)
    mask = lambda s: {"p_gt": s["p_gt"], "mri": s["mri"] * d, "us": s["us"] * d}  # noqa: E731
    bm, bnm, selm = mask(b), mask(bn), mask(sel)
    out = {"fov_diameter_mm": FOV_DIAMETER_MM,
           "full_slices_reference": measure(ctx, ctx.c1, ctx.c2, ctx.c3, b["mri"], b["us"], b["p_gt"]),
           "masked_calibrated_on_full": measure(ctx, ctx.c1, ctx.c2, ctx.c3, bm["mri"], bm["us"], bm["p_gt"])}
    a1, a2 = copy.deepcopy(ctx.c1), copy.deepcopy(ctx.c2)
    reestimate_batchnorm(a1, bnm["mri"])
    reestimate_batchnorm(a2, bnm["us"])
    out["masked_bn_reestimated"] = measure(ctx, a1, a2, ctx.c3, bm["mri"], bm["us"], bm["p_gt"])

    def train_batch(n):
        p = sample_random_6d_poses(n, ctx.shape, ctx.spacing, TRANS_MM, ROT_DEG, device=ctx.device, ref_mm=ctx.ref_mm)
        m = ctx.slices(ctx.vol, p)
        with torch.no_grad():
            u = ctx.simulator(m)
        return p, m * d, u * d

    nets, info = calibrate(ctx, train_batch, bnm["mri"], bnm["us"], selm)
    out["masked_full_calibration"] = measure(ctx, *nets, bm["mri"], bm["us"], bm["p_gt"])
    out["masked_full_calibration"]["calibration"] = info

    # Dose-response: calibrated model and statistics re-estimation for growing field-of-view diameters
    sweep = []
    for dia in (80.0, 120.0, 160.0, 200.0, 240.0, 2 * ctx.ref_mm * math.sqrt(2)):
        dd = fov_disc(ref_mm=ctx.ref_mm, diameter_mm=dia, device=ctx.device)
        s1, s2 = copy.deepcopy(ctx.c1), copy.deepcopy(ctx.c2)
        reestimate_batchnorm(s1, bn["mri"] * dd)
        reestimate_batchnorm(s2, bn["us"] * dd)
        r0 = measure(ctx, ctx.c1, ctx.c2, ctx.c3, b["mri"] * dd, b["us"] * dd, b["p_gt"])
        r1 = measure(ctx, s1, s2, ctx.c3, b["mri"] * dd, b["us"] * dd, b["p_gt"])
        sweep.append({"diameter_mm": dia, "full_slice": dia > 2 * ctx.ref_mm * 1.4,
                      "calibrated_on_full": {p: r0[p]["mte"] for p in ("mri", "us")},
                      "bn_reestimated": {p: r1[p]["mte"] for p in ("mri", "us")}})
    out["fov_sweep"] = sweep
    return out


def part_b(ctx: Context) -> dict:
    ctx.ius = load_ius_volume(volume_options(ctx.cfg)["plane"], ctx.device)
    ev = sample_valid(ctx, 40, seed=42, sim_seed=1234)
    bn = sample_valid(ctx, 160, seed=11, sim_seed=5678)
    sel = sample_valid(ctx, 160, seed=7, sim_seed=4321)
    out = {"poses": {"evaluation_acceptance": ev["acceptance"], "active_px_median": ev["active_px_median"],
                     "trivial": trivial(ctx, ev["p_gt"])},
           "mri_reference": measure(ctx, ctx.c1, ctx.c2, ctx.c3, ev["mri"], ev["sim_full"], ev["p_gt"])["mri"]}
    for key in ("sim_full", "sim_footprint", "real"):
        res = {"calibrated_flair": measure(ctx, ctx.c1, ctx.c2, ctx.c3, ev["mri"], ev[key], ev["p_gt"])}
        if key != "sim_full":
            n2 = copy.deepcopy(ctx.c2)
            reestimate_batchnorm(n2, bn[key])
            res["bn_reestimated"] = measure(ctx, ctx.c1, n2, ctx.c3, ev["mri"], ev[key], ev["p_gt"])
            torch.manual_seed(0)

            def train_batch(n, key=key):
                s = sample_valid(ctx, n, seed=int(torch.randint(0, 2 ** 31 - 1, (1,))), batch=8,
                                 sim_seed=int(torch.randint(0, 2 ** 31 - 1, (1,))))
                return s["p_gt"], s["mri"], s[key]

            nets, info = calibrate(ctx, train_batch, bn["mri"], bn[key], {"p_gt": sel["p_gt"], "mri": sel["mri"],
                                                                           "us": sel[key]})
            res["calibrated_on_input"] = measure(ctx, *nets, ev["mri"], ev[key], ev["p_gt"])
            res["calibrated_on_input"]["calibration"] = info
        out[key] = res
    i = int((ev["real"] > ACTIVE_LEVEL).flatten(1).sum(1).argmax())       # the evaluation plane with most iUS signal
    ctx.example = {k: ev[k][i:i + 1].cpu() for k in ("mri", "sim_full", "real", "p_gt")}
    ctx.example["fov_disc"] = ctx.disc.cpu()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align = align_ius()
    print(f"[*] iUS alignment: header landmark error {align['header_landmark_error_mm']['mean']:.2f} mm; "
          f"rigid fit residual {align['fit_residual_mm']['mean']:.2f} mm, leave-one-out "
          f"{align['leave_one_out_error_mm']['mean']:.2f} mm; NMI {align['nmi_with_flair']}")
    ctx = Context(device)
    a = part_a(ctx)
    b = part_b(ctx)
    b["alignment"] = align

    def line(name, r):
        return (f"  {name:<34} MRI {r['mri']['mte']:6.2f} mm {r['mri']['geo']:5.2f} deg | US {r['us']['mte']:6.2f} mm "
                f"{r['us']['geo']:5.2f} deg | top-1 {r['latent']['top1_retrieval_pct']:5.1f}%")
    print("[*] Part A (central 80 mm field of view, Protocol B)")
    for k in ("full_slices_reference", "masked_calibrated_on_full", "masked_bn_reestimated", "masked_full_calibration"):
        print(line(k, a[k]))
    print(f"[*] Part B (real iUS, {len(ctx.example['p_gt'])}): acceptance {b['poses']['evaluation_acceptance']:.2f}, "
          f"trivial {b['poses']['trivial']}")
    for key in ("sim_full", "sim_footprint", "real"):
        for state, r in b[key].items():
            print(line(f"{key} / {state}", r))
    torch.save(ctx.example, "data/derived/case23_real_ius_example.pt")
    results = load_benchmark()
    results["fov_and_real_ius"] = {"model": str(CAL_CKPT), "stage1": str(UNCAL_CKPT), "fov_mask": a, "real_ius": b}
    BENCHMARK_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[*] fov_and_real_ius written to {BENCHMARK_JSON}")


if __name__ == "__main__":
    main()
