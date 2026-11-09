"""
Directory sweep script for RESECT NIFTI and segmentation datasets.
Identifies case folders, modal volume pairing conventions, and dimensions.
"""

from pathlib import Path
import SimpleITK as sitk


def sweep_resect_dataset():
    nifti_root = Path("data/resect_base/RESECT/NIFTI")
    seg_root = Path("data/jv8bk-osfstorage-archive/RESECT-Segmentation")

    print("=" * 70)
    print("FCN05 DATASET DIRECTORY SWEEP & MODALITY PAIRING AUDIT")
    print("=" * 70)

    if not nifti_root.exists():
        print(f"ERROR: NIFTI root directory not found at {nifti_root.resolve()}")
        return

    # Find and naturally sort Case directories
    cases = sorted(
        [d for d in nifti_root.iterdir() if d.is_dir() and d.name.startswith("Case")],
        key=lambda x: int(x.name.replace("Case", "")) if x.name.replace("Case", "").isdigit() else 999
    )

    print(f"\nFound {len(cases)} Clinical Patient Cases in {nifti_root}:")
    print("-" * 70)
    print(f"{'Case ID':<8} | {'T2-FLAIR MRI File':<24} | {'US Volumes (before/during/after)'}")
    print("-" * 70)

    reader = sitk.ImageFileReader()
    total_pairs = 0

    for case_dir in cases:
        case_id = case_dir.name
        
        # MRI Volume (T2-FLAIR)
        mri_dir = case_dir / "MRI"
        flair_file = mri_dir / f"{case_id}-FLAIR.nii.gz"
        flair_exists = flair_file.exists()
        flair_str = flair_file.name if flair_exists else "MISSING"

        # US Volumes
        us_dir = case_dir / "US"
        us_before = us_dir / f"{case_id}-US-before.nii.gz"
        us_during = us_dir / f"{case_id}-US-during.nii.gz"
        us_after = us_dir / f"{case_id}-US-after.nii.gz"

        us_phases = []
        if us_before.exists():
            us_phases.append("before")
            total_pairs += 1
        if us_during.exists():
            us_phases.append("during")
        if us_after.exists():
            us_phases.append("after")

        print(f"{case_id:<8} | {flair_str:<24} | {', '.join(us_phases)}")

    print("-" * 70)
    print(f"Total Valid (FLAIR <-> US-before) Patient Pairs Available: {total_pairs}")

    # Inspect sample metadata for Case 1
    if len(cases) > 0:
        c1 = cases[0]
        c1_flair = c1 / "MRI" / f"{c1.name}-FLAIR.nii.gz"
        c1_us = c1 / "US" / f"{c1.name}-US-before.nii.gz"

        print("\n" + "=" * 70)
        print(f"SAMPLE VOLUME METADATA INSPECTION ({c1.name}):")
        print("-" * 70)
        
        reader.SetFileName(str(c1_flair))
        reader.ReadImageInformation()
        print(f"Preoperative MRI ({c1_flair.name}):")
        print(f"   Dimensions (X, Y, Z): {reader.GetSize()}")
        print(f"   Voxel Spacing (mm):   {reader.GetSpacing()}")
        print(f"   Origin:               {reader.GetOrigin()}")

        reader.SetFileName(str(c1_us))
        reader.ReadImageInformation()
        print(f"\nIntraoperative US ({c1_us.name}):")
        print(f"   Dimensions (X, Y, Z): {reader.GetSize()}")
        print(f"   Voxel Spacing (mm):   {reader.GetSpacing()}")
        print(f"   Origin:               {reader.GetOrigin()}")

    # Sweep segmentation masks
    print("\n" + "=" * 70)
    print("SEGMENTATION ARCHIVE AUDIT:")
    print("-" * 70)
    if seg_root.exists():
        seg_cases = sorted([d.name for d in seg_root.iterdir() if d.is_dir() and d.name.startswith("Case")])
        print(f"Segmentation folders detected for {len(seg_cases)} cases.")
        sample_seg_files = [f.name for f in (seg_root / seg_cases[0]).iterdir() if not f.name.startswith(".")]
        print(f"Sample structures in {seg_cases[0]}: {sample_seg_files}")
    else:
        print("Segmentation archive not located at expected path.")
    print("=" * 70)


if __name__ == "__main__":
    sweep_resect_dataset()
