"""
Fixed craniotomy centres for the craniotomy-scale Stage 1 networks (Appendix B.4, final design).

For every RESECT patient, the centre is the centroid of the active pre-resection ultrasound signal (> 10% of the 99th
percentile of its non-zero voxels), mapped into the MRI world frame by the least-squares rigid fit to the expert
MRI-iUS landmarks (the tracked header alignment where a patient has no landmark file), and expressed in millimetres
from the centre of the FLAIR volume along (right, anterior, superior) - the (x, y, z) axes of the axial reslicer.
Only the location is used: no ultrasound intensity enters training.
Writes data/derived/craniotomy_centres.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.real_ius_and_fov_probe import read_tags, rigid_fit

NIFTI = Path("data/resect_base/RESECT/NIFTI")
OUT = Path("data/derived/craniotomy_centres.json")


def centre(case: str) -> dict:
    fl = nib.load(str(NIFTI / case / "MRI" / f"{case}-FLAIR.nii.gz"))
    us = nib.load(str(NIFTI / case / "US" / f"{case}-US-before.nii.gz"))
    u = np.asarray(us.dataobj, dtype=np.float32)
    idx = np.argwhere(u > 0.1 * np.percentile(u[u > 0], 99))
    c_us = (us.affine @ np.r_[idx.mean(0), 1.0])[:3]
    tag = NIFTI / case / "Landmarks" / f"{case}-MRI-beforeUS.tag"
    if tag.exists():
        m_pts, u_pts = read_tags(tag)
        r, t = rigid_fit(u_pts, m_pts)
        c_mri, source = r @ c_us + t, "landmark rigid fit"
    else:
        c_mri, source = c_us, "tracked header alignment"
    ras = nib.as_closest_canonical(fl)
    mid = (ras.affine @ np.r_[(np.array(ras.shape) - 1) / 2, 1.0])[:3]
    return {"centre_mm": [float(x) for x in c_mri - mid], "source": source}


def main():
    cases = sorted((p.name for p in NIFTI.iterdir() if p.name.startswith("Case")), key=lambda c: int(c[4:]))
    res = {c: centre(c) for c in cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for c, v in res.items():
        print(f"  {c:<7} {np.round(v['centre_mm'], 1)} ({v['source']})")


if __name__ == "__main__":
    main()
