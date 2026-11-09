"""
Cross-sequence check on the withheld Case 23: does the FLAIR-calibrated primary model localise frames simulated from
the patient's T1-weighted MRI?

1. Alignment. Case23-T1 and Case23-FLAIR share grid size and spacing. Their scanner affines differ by 3.7 mm, but the
   raw voxel grids are already within about one voxel of each other (mutual-information shift search), so the
   headers are not a reliable guide. The primary T1 volume is therefore rigidly registered to FLAIR in voxel space
   (SimpleITK, Euler 3D, Mattes mutual information, three resolutions) and resampled onto the FLAIR grid. For
   transparency the header-affine resampling is evaluated as well.
2. Evaluation. On the Protocol B poses (seed 42, 40 slices, +/-20 mm, +/-30 deg, speckle seed 1234) the calibrated
   S8e model is evaluated on T1-derived inputs (i) with its FLAIR batch-normalisation statistics and (ii) after
   re-estimating them on 160 T1-derived slices (seed 11, speckle 5678), with no weight changes. Both pathways are
   reported: T1 reslices through NN1 and ultrasound simulated from T1 through NN2.
Writes the 'cross_sequence_t1' key of docs/case23_benchmark.json.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_patient import (build_eval_set, per_slice_errors, pipeline_from_config, reestimate_batchnorm,
                                       volume_options)
from scripts.case23_benchmark import BENCHMARK_JSON, CAL_CKPT, load_benchmark, load_pipeline
from scripts.train_pose_pipeline import load_or_create_volume
from src.utils.metrics import pose_unit_mm

MRI_DIR = Path("data/resect_base/RESECT/NIFTI/Case23/MRI")
# Optional: the primary Stage 1 model calibrated on the registered T1 volume, made with
#   python scripts/calibrate_patient.py --case Case23 --data_path data/derived/Case23-T1_registered_to_FLAIR.nii.gz \
#       --save_checkpoint checkpoints/cross_sequence/calibrated_model_case23_t1.pth (same calibration settings as S8e)
T1_CAL_CKPT = Path("checkpoints/cross_sequence/calibrated_model_case23_t1.pth")
DERIVED = {"registered": Path("data/derived/Case23-T1_registered_to_FLAIR.nii.gz"),
           "header_affine": Path("data/derived/Case23-T1_header_affine_to_FLAIR.nii.gz")}


def nmi(a: np.ndarray, b: np.ndarray, mask: np.ndarray, bins: int = 48) -> float:
    """Normalised mutual information (H(A) + H(B)) / H(A, B) over the mask."""
    h, _, _ = np.histogram2d(a[mask], b[mask], bins=bins)
    p = h / h.sum()
    ent = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))  # noqa: E731
    return float((ent(p.sum(1)) + ent(p.sum(0))) / ent(p))


def best_integer_shift(fixed, moving, mask):
    """Integer voxel shift (|s| <= 3) that maximises NMI; (0, 0, 0) means the volumes are aligned to the voxel."""
    best = (nmi(fixed, moving, mask), (0, 0, 0))
    for s in np.ndindex(7, 7, 7):
        shift = tuple(int(x) - 3 for x in s)
        if shift != (0, 0, 0):
            v = nmi(fixed, np.roll(moving, shift, axis=(0, 1, 2)), mask)
            if v > best[0]:
                best = (v, shift)
    return {"nmi_at_zero": nmi(fixed, moving, mask), "best_nmi": best[0], "best_shift_vox": list(best[1])}


def register_rigid(fixed: np.ndarray, moving: np.ndarray):
    """Rigid (Euler 3D) Mattes-MI registration of moving onto fixed in voxel space; returns resampled array, params."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)   # single thread + fixed sampling seed: reproducible result
    f = sitk.GetImageFromArray(fixed)
    m = sitk.GetImageFromArray(moving)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10, seed=1234)   # SimpleITK treats seed 0 as "wall clock"
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300,
                                                 relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY))
    tx = reg.Execute(f, m)
    out = sitk.GetArrayFromImage(sitk.Resample(m, f, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    params = sitk.Euler3DTransform(tx.GetNthTransform(0) if tx.GetName() == "CompositeTransform" else tx).GetParameters()
    return out, {"rotation_deg": [math.degrees(a) for a in params[:3]], "translation_vox": list(params[3:]),
                 "final_metric": reg.GetMetricValue(), "iterations": reg.GetOptimizerIteration()}


def prepare_t1() -> dict:
    fl_img = nib.load(str(MRI_DIR / "Case23-FLAIR.nii.gz"))
    t1_img = nib.load(str(MRI_DIR / "Case23-T1.nii.gz"))
    fl, t1 = fl_img.get_fdata(dtype=np.float32), t1_img.get_fdata(dtype=np.float32)
    mask = fl > np.percentile(fl, 60)
    m = np.linalg.inv(t1_img.affine) @ fl_img.affine
    header = ndimage.affine_transform(t1, m[:3, :3], m[:3, 3], output_shape=fl.shape, order=1, cval=0.0)
    registered, params = register_rigid(fl, t1)
    for key, vol in (("registered", registered), ("header_affine", header)):
        DERIVED[key].parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(vol.astype(np.float32), fl_img.affine, fl_img.header), str(DERIVED[key]))
    return {"flair_shape": list(fl.shape), "t1_shape": list(t1.shape),
            "flair_spacing_mm": [float(z) for z in fl_img.header.get_zooms()[:3]],
            "t1_spacing_mm": [float(z) for z in t1_img.header.get_zooms()[:3]],
            "affine_translation_difference_mm": float(np.linalg.norm((fl_img.affine - t1_img.affine)[:3, 3])),
            "raw_grid": best_integer_shift(fl, t1, mask),
            "header_affine": best_integer_shift(fl, header, mask),
            "registered": best_integer_shift(fl, registered, mask),
            "registration": params}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align = prepare_t1()
    for k in ("raw_grid", "header_affine", "registered"):
        a = align[k]
        print(f"[*] {k:<14} NMI {a['nmi_at_zero']:.4f} at zero shift; best {a['best_nmi']:.4f} at {a['best_shift_vox']}")
    r = align["registration"]
    print(f"[*] rigid registration: rotation {[round(x, 2) for x in r['rotation_deg']]} deg, "
          f"translation {[round(x, 2) for x in r['translation_vox']]} vox")

    (nn1, nn2, nn3), ckpt = load_pipeline(CAL_CKPT, device)
    cfg = ckpt.get("config")
    reslicer, simulator, ref_mm, *_ = pipeline_from_config(cfg, device)

    def measure(n1, n2, s, unit, n3=None):
        n3 = nn3 if n3 is None else n3
        out = {}
        with torch.no_grad():
            for branch, enc, img in (("mri", n1, s["mri"]), ("us", n2, s["us"])):
                e = per_slice_errors(n3(enc(img)), s["p_gt"], unit)
                out[branch] = {k: float(v.mean()) for k, v in e.items()}          # mte, geo, tx, ty, tz, pitch, roll, yaw
                out[branch].update({"median_mte": float(np.median(e["mte"])), "max_mte": float(e["mte"].max())})
        return out

    def evaluate(path):
        vol, shape, spacing = load_or_create_volume(path, device, **volume_options(cfg))
        unit = pose_unit_mm(shape, spacing, ref_mm)
        s_b = build_eval_set(reslicer, vol, shape, spacing, 10, 4, 20.0, 30.0, seed=42, simulator=simulator,
                             sim_seed=1234, device=device)
        bn_set = build_eval_set(reslicer, vol, shape, spacing, 40, 4, 20.0, 30.0, seed=11, simulator=simulator,
                                sim_seed=5678, device=device)
        b1, b2 = copy.deepcopy(nn1), copy.deepcopy(nn2)
        reestimate_batchnorm(b1, bn_set["mri"])
        reestimate_batchnorm(b2, bn_set["us"])
        out = {"flair_bn_stats": measure(nn1, nn2, s_b, unit), "bn_reestimated": measure(b1, b2, s_b, unit)}
        if Path(path) == DERIVED["registered"] and T1_CAL_CKPT.exists():
            (t1, t2, t3), _ = load_pipeline(T1_CAL_CKPT, device)
            out["full_calibration_on_t1"] = measure(t1, t2, s_b, unit, n3=t3)
        return s_b["p_gt"], out

    p_fl, flair = evaluate(MRI_DIR / "Case23-FLAIR.nii.gz")
    res = {"alignment": align, "model": str(CAL_CKPT), "flair_reference": flair["flair_bn_stats"],
           "flair_bn_reestimated_check": flair["bn_reestimated"]}
    for key in ("registered", "header_affine"):
        p_t1, out = evaluate(DERIVED[key])
        assert torch.equal(p_fl, p_t1), "Protocol B poses must be identical for both sequences"
        res[f"t1_{key}"] = out
    res["primary"] = "t1_registered"

    rows = [("FLAIR (reference)", res["flair_reference"]), ("FLAIR, BN re-est.", res["flair_bn_reestimated_check"])]
    for key in ("registered", "header_affine"):
        rows += [(f"T1 {key}, FLAIR BN", res[f"t1_{key}"]["flair_bn_stats"]),
                 (f"T1 {key}, T1 BN", res[f"t1_{key}"]["bn_reestimated"])]
        if "full_calibration_on_t1" in res[f"t1_{key}"]:
            rows.append((f"T1 {key}, calibrated on T1", res[f"t1_{key}"]["full_calibration_on_t1"]))
    for name, r in rows:
        print(f"  {name:<28} MRI {r['mri']['mte']:5.2f} mm {r['mri']['geo']:5.2f} deg | "
              f"Sim-iUS {r['us']['mte']:5.2f} mm {r['us']['geo']:5.2f} deg")
    results = load_benchmark()
    results["cross_sequence_t1"] = res
    BENCHMARK_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[*] cross_sequence_t1 written to {BENCHMARK_JSON}")


if __name__ == "__main__":
    main()
