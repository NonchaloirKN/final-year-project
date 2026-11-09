"""
Additional leave-one-subject-out folds for the primary (S8e) pipeline.

For each withheld patient this runs exactly the primary configuration:
  Stage 1  - 100 epochs x 25 steps x 16, AdamW 3e-4, ImageNet initialisation, AMP, seed 0, axial canonical plane,
             millimetre-rigid geometry, five-stage simulator, contrastive weight 0.1, 50-epoch rotation curriculum;
  Stage 2a - BatchNorm re-estimation on 160 pre-operative slices (seed 11);
  Stage 2b - 10 epochs of NN3 + projection-head fine-tuning (variant chosen on the seed-7 selection set).
Each fold is written to checkpoints/loso_folds/<case>/, so the primary Case 23 checkpoints and the thesis figures
(docs/sprint_4_initial_training_curves.png, docs/sprint_5_calibration_curves.png) are never overwritten.

Usage (one stage per call keeps each run short and its memory freed when the process exits):
    python scripts/run_loso_folds.py --stage train --cases Case1
    python scripts/run_loso_folds.py --stage calibrate --cases Case1
    python scripts/run_loso_folds.py --stage evaluate --cases Case23 Case1 Case12 Case2

The evaluate stage measures every listed patient on its own Protocol B set (seed 42, 40 slices, +/-20 mm, +/-30 deg,
speckle seed 1234) in three states - uncalibrated, BatchNorm re-estimation only, fully calibrated - plus the
uncalibrated top-1 cross-modal retrieval, and writes the 'loso_4fold_summary' key of docs/case23_benchmark.json.
Case 23 uses the primary checkpoints in checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLD_ROOT = Path("checkpoints/loso_folds")
PRIMARY = {"uncal": Path("checkpoints/best_model_pose_pipeline.pth"),
           "cal": Path("checkpoints/calibrated_model_case23.pth")}

TRAIN_ARGS = ["--from_scratch", "--pretrained", "--epochs", "100", "--steps_per_epoch", "25", "--batch_size", "16",
              "--lr", "3e-4", "--contrastive_weight", "0.1", "--rot_curriculum_epochs", "50", "--amp", "--seed", "0",
              "--quiet", "--log_every", "10", "--vis_freq", "100"]
CALIB_ARGS = ["--epochs", "10", "--steps_per_epoch", "25", "--batch_size", "16", "--lr", "3e-4",
              "--variant", "compare", "--bn_reestimate", "--seed", "0", "--quiet"]


def fold_paths(case: str):
    if case == "Case23":
        return PRIMARY["uncal"], PRIMARY["cal"]
    d = FOLD_ROOT / case.lower()
    return d / "best_model_pose_pipeline.pth", d / f"calibrated_model_{case.lower()}.pth"


def run(cmd):
    print("[*] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def train(case: str):
    uncal, _ = fold_paths(case)
    uncal.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "scripts/train_pose_pipeline.py", "--val_case", case, "--save_checkpoint", str(uncal),
         "--output_plot", str(uncal.parent / "training_curves.png")] + TRAIN_ARGS)


def calibrate(case: str):
    uncal, cal = fold_paths(case)
    run([sys.executable, "scripts/calibrate_patient.py", "--case", case, "--checkpoint", str(uncal),
         "--save_checkpoint", str(cal), "--output_plot", str(cal.parent / "calibration_curves.png")] + CALIB_ARGS)


def evaluate(cases):
    import copy

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from scripts.calibrate_patient import (build_eval_set, find_patient_volume, per_slice_errors,
                                           pipeline_from_config, reestimate_batchnorm, volume_options)
    from scripts.case23_benchmark import BENCHMARK_JSON, cosine_diagnostics, load_benchmark, load_pipeline
    from scripts.train_pose_pipeline import load_or_create_volume
    from src.utils.metrics import pose_unit_mm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = {}
    for case in cases:
        uncal_path, cal_path = fold_paths(case)
        (nn1, nn2, nn3), ckpt = load_pipeline(uncal_path, device)
        cfg = ckpt.get("config")
        assert (cfg or {}).get("val_case") == case, f"{uncal_path} was trained with val_case {cfg.get('val_case')}"
        vol, shape, spacing = load_or_create_volume(find_patient_volume(case), device, **volume_options(cfg))
        reslicer, simulator, ref_mm, *_ = pipeline_from_config(cfg, device)
        unit = pose_unit_mm(shape, spacing, ref_mm)
        s_b = build_eval_set(reslicer, vol, shape, spacing, 10, 4, 20.0, 30.0, seed=42, simulator=simulator,
                             sim_seed=1234, device=device)
        bn_set = build_eval_set(reslicer, vol, shape, spacing, 40, 4, 20.0, 30.0, seed=11, simulator=simulator,
                                sim_seed=5678, device=device)

        def measure(n1, n2, n3):
            out = {}
            with torch.no_grad():
                for branch, enc, img in (("mri", n1, s_b["mri"]), ("us", n2, s_b["us"])):
                    e = per_slice_errors(n3(enc(img)), s_b["p_gt"], unit)
                    out[branch] = {"mte": float(e["mte"].mean()), "geo": float(e["geo"].mean()),
                                   "median_mte": float(np.median(e["mte"])), "tz": float(e["tz"].mean())}
            return out

        rec = {"volume_shape": list(shape), "spacing_mm": list(spacing), "uncalibrated": measure(nn1, nn2, nn3)}
        with torch.no_grad():
            rec["top1_retrieval_pct"] = cosine_diagnostics(nn1(s_b["mri"]), nn2(s_b["us"]))["top1_retrieval_pct"]
        b1, b2 = copy.deepcopy(nn1), copy.deepcopy(nn2)
        reestimate_batchnorm(b1, bn_set["mri"])
        reestimate_batchnorm(b2, bn_set["us"])
        rec["bn_only"] = measure(b1, b2, nn3)
        (c1, c2, c3), cal_ckpt = load_pipeline(cal_path, device)
        rec["calibrated"] = measure(c1, c2, c3)
        rec["calibration"] = {"variant": cal_ckpt.get("variant"), "best_epoch": cal_ckpt.get("best_epoch"),
                              "seconds": (cal_ckpt.get("history") or {}).get("calibration_seconds", [None])[0]}
        rec["stage1_epoch"] = ckpt.get("epoch")
        folds[case] = rec
        print(f"  {case:<7} top-1 {rec['top1_retrieval_pct']:5.1f}% | " + " | ".join(
            f"{st}: MRI {rec[st]['mri']['mte']:5.2f} mm {rec[st]['mri']['geo']:4.2f} deg, "
            f"US {rec[st]['us']['mte']:5.2f} mm {rec[st]['us']['geo']:4.2f} deg"
            for st in ("uncalibrated", "bn_only", "calibrated")), flush=True)
        del nn1, nn2, nn3, b1, b2, c1, c2, c3, vol, s_b, bn_set
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {}
    for st in ("uncalibrated", "bn_only", "calibrated"):
        for br in ("mri", "us"):
            for m in ("mte", "geo"):
                v = np.array([folds[c][st][br][m] for c in cases])
                summary[f"{st}_{br}_{m}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    v = np.array([folds[c]["top1_retrieval_pct"] for c in cases])
    summary["top1_retrieval_pct"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    results = load_benchmark()
    results["loso_4fold_summary"] = {
        "cases": list(cases), "protocol": {"seed": 42, "slices": 40, "trans_mm": 20.0, "rot_deg": 30.0, "speckle_seed": 1234},
        "bn_set": {"seed": 11, "slices": 160, "speckle_seed": 5678}, "folds": folds, "mean_sd": summary,
        "note": "Case23 uses the primary S8e checkpoints; the other folds were trained and calibrated with "
                "scripts/run_loso_folds.py using the identical configuration."}
    BENCHMARK_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("  mean +/- SD over folds: " + ", ".join(
        f"{k} {v['mean']:.2f} +/- {v['sd']:.2f}" for k, v in summary.items() if k.endswith("_mte") or k.startswith("top1")))
    print(f"[*] loso_4fold_summary written to {BENCHMARK_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "calibrate", "evaluate"], required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    args = ap.parse_args()
    if args.stage == "evaluate":
        evaluate(args.cases)
        return
    for case in args.cases:
        if case == "Case23":
            raise SystemExit("Case23 is the primary fold; its checkpoints already exist in checkpoints/")
        (train if args.stage == "train" else calibrate)(case)


if __name__ == "__main__":
    main()
