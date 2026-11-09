"""
Case 23 True-Millimetre Benchmark (Sprint 8, Tasks S8.6d / S8.7b).

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Single source of truth for every Chapter 4/5 number and figure:
1. Protocol A: seed 42, 10 x 4 slices, +/-20 mm, +/-15 deg (the calibration trajectory set).
2. Protocol B: seed 42, 10 x 4 slices, +/-20 mm, +/-30 deg (wide-perturbation distribution).
   Simulated iUS frames are synthesised from each MRI reslice (speckle seed 1234, per batch of 4).
3. Both pathways, uncalibrated vs zero-US calibrated:
   - MRI     -> NN1 -> NN3
   - Sim-iUS -> NN2 -> NN3
4. Trivial baseline: volume-centre translation + identity rotation.
5. Latent diagnostics: matched / mismatched MRI-US cosine, inter-slice cosine, paired L2, top-1 retrieval.
6. Single-frame latency of Sim-iUS -> NN2 -> NN3 on the local GPU.
7. Earlier models for comparison, each on the pipeline it was trained with (rebuilt from its checkpoint
   config): Sprint 7 Colab and Sprint 8a (legacy geometry and simulator).
Translations are reported in true millimetres (pose_unit_mm).
Writes docs/case23_benchmark.json (summaries + per-slice Protocol B errors) and prints the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.calibrate_patient import build_eval_set, per_slice_errors, pipeline_from_config, summarise_errors, volume_options
from scripts.train_pose_pipeline import load_or_create_volume
from src.models.encoders import NN1, NN2
from src.models.pose_regressor import NN3
from src.utils.metrics import pose_unit_mm, volume_half_extent_mm

CASE23_PATH = Path("data/resect_base/RESECT/NIFTI/Case23/MRI/Case23-FLAIR.nii.gz")
UNCAL_CKPT = Path("checkpoints/best_model_pose_pipeline.pth")
CAL_CKPT = Path("checkpoints/calibrated_model_case23.pth")
ARCHIVES = {"legacy": Path("checkpoints/sprint7_colab"), "s8a": Path("checkpoints/sprint8a"),
            "s8b": Path("checkpoints/sprint8b"), "s8c": Path("checkpoints/s8c_candidate"),
            "s8c060": Path("checkpoints/s8c_alpha060"), "s8b_nocurr": Path("checkpoints/ablations/phys_no_curriculum"),
            "s8d": Path("checkpoints/s8d_axial"), "bnonly": Path("checkpoints/s8e_bn_only")}
BENCHMARK_JSON = Path("docs/case23_benchmark.json")
PROTOCOLS = {"A": 15.0, "B": 30.0}


def load_pipeline(ckpt_path: Path, device: torch.device):
    """Load NN1, NN2, NN3 (trunk variant from the checkpoint config) in eval mode."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    symmetric = bool((ckpt.get("config") or {}).get("symmetric_trunk", False))
    nets = (NN1(embedding_dim=128).to(device), NN2(embedding_dim=128).to(device),
            NN3(in_features=128, symmetric_trunk=symmetric).to(device))
    for net, key in zip(nets, ("nn1_state_dict", "nn2_state_dict", "nn3_state_dict")):
        net.load_state_dict(ckpt[key])
        net.eval()
    return nets, ckpt


def build_protocol_sets(vol, shape, spacing, reslicer, simulator, device) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: build_eval_set(reslicer, vol, shape, spacing, 10, 4, 20.0, rot, seed=42,
                                 simulator=simulator, sim_seed=1234, device=device)
            for name, rot in PROTOCOLS.items()}


def load_case23(device: torch.device, ckpt_path: Path = UNCAL_CKPT):
    """
    Case 23 volume and the Protocol A/B sets for the pipeline recorded in ckpt_path.

    Returns (vol, shape, spacing, unit_mm, sets, reslicer, simulator, extent_mm), where unit_mm converts pose
    translations to mm and extent_mm is the volume half-extent the reslicer needs.
    """
    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("config")
    vol, shape, spacing = load_or_create_volume(CASE23_PATH, device, **volume_options(cfg))
    reslicer, simulator, ref_mm, *_ = pipeline_from_config(cfg, device)
    sets = build_protocol_sets(vol, shape, spacing, reslicer, simulator, device)
    return (vol, shape, spacing, pose_unit_mm(shape, spacing, ref_mm), sets, reslicer, simulator,
            volume_half_extent_mm(shape, spacing))


def cosine_diagnostics(z_mri: torch.Tensor, z_us: torch.Tensor) -> Dict[str, float]:
    """Matched vs mismatched cross-modal cosine, inter-slice cosine within each modality, paired L2."""
    z_mri = F.normalize(z_mri, dim=-1)
    z_us = F.normalize(z_us, dim=-1)
    n = z_mri.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=z_mri.device)
    cross = z_mri @ z_us.T
    matched = torch.diagonal(cross)
    return {
        "matched_cos_mean": float(matched.mean()),
        "matched_cos_std": float(matched.std()),
        "mismatched_cos_mean": float(cross[off].mean()),
        "inter_slice_us_cos": float((z_us @ z_us.T)[off].mean()),
        "inter_slice_mri_cos": float((z_mri @ z_mri.T)[off].mean()),
        "paired_l2": float((z_mri - z_us).norm(dim=-1).mean()),
        "top1_retrieval_pct": float((cross.argmax(dim=1) == torch.arange(n, device=z_mri.device)).float().mean() * 100),
    }


def measure_latency_ms(nn2, nn3, frame: torch.Tensor, reps: int = 200) -> float:
    """Median single-frame NN2 -> NN3 latency in ms (CUDA-synchronised)."""
    sync = torch.cuda.synchronize if frame.is_cuda else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(reps + 20):
            sync()
            t0 = time.perf_counter()
            nn3(nn2(frame))
            sync()
            if i >= 20:
                times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def evaluate_model(nets, sets, unit) -> Tuple[Dict, Dict]:
    """Summaries for both pathways on every protocol, plus Protocol B per-slice errors and embeddings."""
    nn1, nn2, nn3 = nets
    summary, per_slice = {}, {}
    with torch.no_grad():
        for name, s in sets.items():
            for branch, enc, img in (("mri", nn1, s["mri"]), ("us", nn2, s["us"])):
                errs = per_slice_errors(nn3(enc(img)), s["p_gt"], unit)
                summary[f"{name}_{branch}"] = summarise_errors(errs)
                summary[f"{name}_{branch}"]["pct_under_5mm"] = float(np.mean(errs["mte"] < 5.0) * 100)
                summary[f"{name}_{branch}"]["pct_under_10mm"] = float(np.mean(errs["mte"] < 10.0) * 100)
                summary[f"{name}_{branch}"]["pct_under_2mm"] = float(np.mean(errs["mte"] < 2.0) * 100)
                summary[f"{name}_{branch}"]["pct_under_3mm"] = float(np.mean(errs["mte"] < 3.0) * 100)
                summary[f"{name}_{branch}"]["median_mte"] = float(np.median(errs["mte"]))
                summary[f"{name}_{branch}"]["median_geo"] = float(np.median(errs["geo"]))
                if name == "B":
                    per_slice[branch] = {k: v.tolist() for k, v in errs.items()}
        summary["latent_B"] = cosine_diagnostics(nn1(sets["B"]["mri"]), nn2(sets["B"]["us"]))
    return summary, per_slice


def run_benchmark(device: torch.device, include_legacy: bool = True) -> Dict:
    vol, shape, spacing = load_or_create_volume(CASE23_PATH, device)
    vol_cache: Dict[str, Tuple] = {}
    models = {"uncalibrated": UNCAL_CKPT, "calibrated": CAL_CKPT}
    if include_legacy:
        for prefix, folder in ARCHIVES.items():
            if (folder / UNCAL_CKPT.name).exists():
                models[f"{prefix}_uncalibrated"] = folder / UNCAL_CKPT.name
                models[f"{prefix}_calibrated"] = folder / CAL_CKPT.name

    results: Dict = {"case": "Case23", "volume_shape": list(shape), "spacing_mm": list(spacing),
                     "half_extent_mm": list(volume_half_extent_mm(shape, spacing)),
                     "protocols": {k: {"seed": 42, "slices": 40, "trans_mm": 20.0, "rot_deg": v} for k, v in PROTOCOLS.items()},
                     "models": {}, "per_slice_B": {}, "pipelines": {}}
    set_cache: Dict[str, Tuple] = {}
    for tag, path in models.items():
        nets, ckpt = load_pipeline(path, device)
        cfg = ckpt.get("config")
        reslicer, simulator, ref_mm, *_ = pipeline_from_config(cfg, device)
        vopts = volume_options(cfg)
        vkey = json.dumps(vopts, sort_keys=True)
        if vkey not in vol_cache:
            vol_cache[vkey] = load_or_create_volume(CASE23_PATH, device, **vopts)
        vol, shape, spacing = vol_cache[vkey]
        key = json.dumps({"ref": ref_mm, "sim": simulator.config(), "vol": vopts}, sort_keys=True)
        if key not in set_cache:
            set_cache[key] = (build_protocol_sets(vol, shape, spacing, reslicer, simulator, device),
                              pose_unit_mm(shape, spacing, ref_mm))
        sets, unit = set_cache[key]
        results["pipelines"][tag] = {"geometry": "physical" if ref_mm else "legacy", "ref_mm": ref_mm, **vopts,
                                     "simulator": simulator.config(), "symmetric_trunk": nets[2].symmetric_trunk}
        summary, per_slice = evaluate_model(nets, sets, unit)
        results["models"][tag] = summary
        if tag in ("uncalibrated", "calibrated", "s8c060_calibrated", "s8d_calibrated"):
            results["per_slice_B"][tag] = per_slice
        if tag == "uncalibrated":
            p_gt = sets["B"]["p_gt"]
            results["gt_tz_mm_B"] = (p_gt[:, 2] * unit[2]).cpu().tolist()
            trivial = {}
            for name, s in sets.items():
                p0 = torch.zeros_like(s["p_gt"])
                p0[:, 3] = 1.0
                p0[:, 7] = 1.0
                trivial[name] = summarise_errors(per_slice_errors(p0, s["p_gt"], unit))
            results["trivial"] = trivial
            hist = ckpt.get("history") or {}
            results["training"] = {"config": cfg, "epoch": ckpt.get("epoch"),
                                   "final_val_mte_mri_mm": hist.get("val_mte_mri_mm", [None])[-1],
                                   "final_val_mte_us_mm": hist.get("val_mte_us_mm", [None])[-1]}
        if tag == "calibrated":
            hist = ckpt.get("history", {})
            results["calibration"] = {
                "best_epoch": hist.get("best_epoch", [None])[0],
                "seconds": hist.get("calibration_seconds", [None])[0],
                "branches": ckpt.get("branches"),
                "variant": ckpt.get("variant"),
                "variant_selection_loss": hist.get("variant_selection_loss"),
                "trajectory_A": {k: hist[k] for k in ("epoch", "val_mte_mm", "val_mae_deg", "val_us_mte_mm", "val_us_mae_deg",
                                                      "val_tz_err_mm", "val_us_tz_err_mm", "selection_loss") if k in hist},
                "variant_trajectories_A": {v: {k: h[k] for k in ("val_mte_mm", "val_us_mte_mm", "selection_loss", "best_epoch")}
                                           for v, h in (hist.get("variant_histories") or {}).items()},
                "config": (ckpt.get("config") or {}).get("calibration"),
            }
            results["latency_ms_nn2_nn3"] = measure_latency_ms(nets[1], nets[2], sets["B"]["us"][:1])
        del nets
    return results


POSE_SEEDS = (42, 43, 44, 45, 46)
SPECKLE_SEEDS = (1234, 1235, 1236, 1237, 1238)


def bootstrap_ci(x: np.ndarray, n_resamples: int = 10_000, level: float = 95.0, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_resamples, len(x)), replace=True).mean(axis=1)
    tail = (100.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def robustness_analysis(device: torch.device) -> Dict:
    """
    Statistical robustness of the primary pipeline (uncalibrated and calibrated S8e checkpoints) on Case 23.

    1. Bootstrap: 10,000-resample percentile 95% CIs of the mean MTE and geodesic error, and the worst slice,
       on the 40-slice Protocol A and B sets (pose seed 42, speckle seed 1234).
    2. Multi-seed: Protocol B (+/-20 mm, +/-30 deg) on every combination of 5 pose seeds x 5 speckle seeds
       (200 distinct poses). The MRI pathway does not depend on the speckle seed.
    """
    cfg = torch.load(UNCAL_CKPT, map_location="cpu", weights_only=False).get("config")
    vol, shape, spacing = load_or_create_volume(CASE23_PATH, device, **volume_options(cfg))
    reslicer, simulator, ref_mm, *_ = pipeline_from_config(cfg, device)
    unit = pose_unit_mm(shape, spacing, ref_mm)
    nets = {tag: load_pipeline(path, device)[0] for tag, path in (("uncalibrated", UNCAL_CKPT), ("calibrated", CAL_CKPT))}

    def errors(nets_, s, branch):
        nn1, nn2, nn3 = nets_
        with torch.no_grad():
            pred = nn3(nn1(s["mri"])) if branch == "mri" else nn3(nn2(s["us"]))
        return per_slice_errors(pred, s["p_gt"], unit)

    out: Dict = {"bootstrap": {"resamples": 10_000, "level_pct": 95, "pose_seed": 42, "speckle_seed": 1234},
                 "multiseed_B": {"pose_seeds": list(POSE_SEEDS), "speckle_seeds": list(SPECKLE_SEEDS),
                                 "slices_per_set": 40, "trans_mm": 20.0, "rot_deg": PROTOCOLS["B"]}}
    for proto, rot in PROTOCOLS.items():
        s = build_eval_set(reslicer, vol, shape, spacing, 10, 4, 20.0, rot, seed=42, simulator=simulator,
                           sim_seed=1234, device=device)
        for tag, nets_ in nets.items():
            for branch in ("mri", "us"):
                e = errors(nets_, s, branch)
                rec = {}
                for metric in ("mte", "geo"):
                    lo, hi = bootstrap_ci(e[metric])
                    rec[metric] = {"mean": float(e[metric].mean()), "ci95": [lo, hi], "sd": float(e[metric].std(ddof=1)),
                                   "max": float(e[metric].max())}
                out["bootstrap"][f"{tag}_{proto}_{branch}"] = rec

    grid: Dict[str, Dict[str, List[List[float]]]] = {tag: {f"{b}_{m}": [] for b in ("mri", "us") for m in ("mte", "geo")}
                                                     for tag in nets}
    for ps in POSE_SEEDS:
        rows = {tag: {k: [] for k in grid[tag]} for tag in nets}
        for ss in SPECKLE_SEEDS:
            s = build_eval_set(reslicer, vol, shape, spacing, 10, 4, 20.0, PROTOCOLS["B"], seed=ps, simulator=simulator,
                               sim_seed=ss, device=device)
            for tag, nets_ in nets.items():
                for branch in ("mri", "us"):
                    e = errors(nets_, s, branch)
                    rows[tag][f"{branch}_mte"].append(float(e["mte"].mean()))
                    rows[tag][f"{branch}_geo"].append(float(e["geo"].mean()))
        for tag in nets:
            for k in grid[tag]:
                grid[tag][k].append(rows[tag][k])
    for tag in nets:
        summ = {}
        for k, g in grid[tag].items():
            g = np.array(g)                                   # [pose seed, speckle seed]
            summ[k] = {"grand_mean": float(g.mean()),
                       "pose_seed_means": g.mean(axis=1).tolist(), "sd_across_pose_seeds": float(g.mean(axis=1).std(ddof=1)),
                       "speckle_seed_means": g.mean(axis=0).tolist(), "sd_across_speckle_seeds": float(g.mean(axis=0).std(ddof=1)),
                       "min_set_mean": float(g.min()), "max_set_mean": float(g.max()),
                       "sd_all_sets": float(g.std(ddof=1)),
                       "mean_within_pose_seed_sd": float(g.std(axis=1, ddof=1).mean()),   # speckle effect at fixed poses
                       "grid": g.tolist()}
        out["multiseed_B"][tag] = summ
    return out


def print_robustness(rob: Dict) -> None:
    print("\nBootstrap 95% CI (pose seed 42, speckle 1234): mean [lo, hi] | worst slice")
    for key, rec in rob["bootstrap"].items():
        if isinstance(rec, dict) and "mte" in rec:
            m, g = rec["mte"], rec["geo"]
            print(f"  {key:<20} MTE {m['mean']:5.2f} [{m['ci95'][0]:5.2f}, {m['ci95'][1]:5.2f}] max {m['max']:5.2f} mm | "
                  f"geo {g['mean']:5.2f} [{g['ci95'][0]:5.2f}, {g['ci95'][1]:5.2f}] max {g['max']:5.2f} deg")
    print("Protocol B, 5 pose seeds x 5 speckle seeds: grand mean | SD across pose seeds | SD across speckle seeds")
    for tag in ("uncalibrated", "calibrated"):
        for k, v in rob["multiseed_B"][tag].items():
            print(f"  {tag:<12} {k:<8} {v['grand_mean']:6.2f} | {v['sd_across_pose_seeds']:5.2f} | "
                  f"{v['sd_across_speckle_seeds']:5.2f} | set means {v['min_set_mean']:.2f}-{v['max_set_mean']:.2f}")


def print_table(r: Dict) -> None:
    def row(label, s):
        return (f"{label:<24}| {s['mte']:6.2f} | {s['tx']:5.2f} {s['ty']:5.2f} {s['tz']:5.2f} | "
                f"{s['geo']:5.2f} | {s['pitch']:5.2f} {s['roll']:5.2f} {s['yaw']:5.2f}")
    for proto in ("A", "B"):
        print(f"\nProtocol {proto} (+/-{PROTOCOLS[proto]:.0f} deg) | MTE | Tx Ty Tz | MAE | Pitch Roll Yaw")
        print(row("Trivial (centre, I)", r["trivial"][proto]))
        for tag in ("legacy_uncalibrated", "legacy_calibrated", "s8a_uncalibrated", "s8a_calibrated",
                    "uncalibrated", "calibrated"):
            if tag not in r["models"]:
                continue
            short = tag.replace("legacy_", "S7 ").replace("s8a_", "S8a ").replace("uncalibrated", "unc").replace("calibrated", "cal")
            for branch, name in (("mri", "MRI"), ("us", "US")):
                print(row(f"{name} {short}", r["models"][tag][f"{proto}_{branch}"]))
    print("\nLatent B: matched / mismatched / inter-US / inter-MRI cos | top-1 %")
    for tag in ("legacy_uncalibrated", "s8a_uncalibrated", "uncalibrated"):
        if tag in r["models"]:
            c = r["models"][tag]["latent_B"]
            print(f"  {tag:<20} {c['matched_cos_mean']:.3f} / {c['mismatched_cos_mean']:.3f} / "
                  f"{c['inter_slice_us_cos']:.3f} / {c['inter_slice_mri_cos']:.3f} | {c['top1_retrieval_pct']:.1f}")
    cal = r.get("calibration", {})
    print(f"Calibration: variant {cal.get('variant')} {cal.get('variant_selection_loss')}, epoch {cal.get('best_epoch')}, "
          f"{cal.get('seconds', 0):.1f} s; latency {r['latency_ms_nn2_nn3']:.2f} ms/frame")


def load_benchmark(path: Path = BENCHMARK_JSON) -> Dict:
    """Load the benchmark JSON written by this script (figure scripts read from here)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/case23_benchmark.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Case 23 true-millimetre dual-pathway benchmark")
    parser.add_argument("--no_legacy", action="store_true", help="Skip the archived Sprint 7 / 8a checkpoints")
    parser.add_argument("--out", type=Path, default=BENCHMARK_JSON)
    parser.add_argument("--robustness", action="store_true",
                        help="Only (re)compute the bootstrap CIs and the 5 x 5 seed analysis and merge them into the "
                             "existing benchmark JSON, leaving every other entry unchanged")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.robustness:
        results = load_benchmark(args.out)
        results["robustness"] = robustness_analysis(device)
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print_robustness(results["robustness"])
        print(f"\n[*] Robustness analysis merged into {args.out}")
        return
    results = run_benchmark(device, include_legacy=not args.no_legacy)
    results["robustness"] = robustness_analysis(device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print_table(results)
    print(f"\n[*] Benchmark written to {args.out}")


if __name__ == "__main__":
    main()
