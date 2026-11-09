"""
Latent feature PCA visualiser: joint channel-PCA of the NN1/NN2 feature maps.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Procedure:
1. Ingests a paired clean MRI slice and synthetic ultrasound slice.
2. Extracts the pre-GAP 512-channel spatial feature maps from NN1 (MRI) and NN2 (US).
3. Projects the 512 latent channels down to 3 principal components (RGB) via joint PCA.
4. Normalises to [0, 255] and visualises the modality-invariant 'primal sketch' as an RGB image.
5. Saves publication-grade side-by-side comparison figure to figures/latent_feature_pca.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from scripts.train_pose_pipeline import load_or_create_volume
from src.models.encoders import NN1, NN2, MRIEncoder, USEncoder
from src.simulator.sim_us import AcousticSimulator
from src.utils.reslicer import VolumeReslicer


def project_features_to_rgb(
    feat_mri: torch.Tensor,
    feat_us: torch.Tensor,
    target_size: Tuple[int, int] = (227, 227),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Project 512-D spatial feature maps of MRI and US to 3-channel RGB images via joint PCA.

    Args:
        feat_mri: [1, 512, H_f, W_f] feature map from NN1.
        feat_us:  [1, 512, H_f, W_f] feature map from NN2.
        target_size: (H, W) spatial dimensions to upsample feature maps to.

    Returns:
        (rgb_mri, rgb_us, explained_variance_ratio):
            rgb_mri: [H, W, 3] uint8 image array.
            rgb_us:  [H, W, 3] uint8 image array.
            explained_variance: cumulative variance explained by top 3 components.
    """
    # 1. Bilinearly upsample spatial features to slice dimensions
    feat_mri_up = F.interpolate(feat_mri, size=target_size, mode="bilinear", align_corners=False)
    feat_us_up = F.interpolate(feat_us, size=target_size, mode="bilinear", align_corners=False)

    # 2. Reshape to [N_pixels, 512]
    h, w = target_size
    mri_flat = feat_mri_up.squeeze(0).permute(1, 2, 0).reshape(-1, 512).detach().cpu().numpy()
    us_flat = feat_us_up.squeeze(0).permute(1, 2, 0).reshape(-1, 512).detach().cpu().numpy()

    # 3. Fit joint PCA across both modalities for consistent color-feature correspondence
    combined = np.vstack([mri_flat, us_flat])
    pca = PCA(n_components=3, random_state=42)
    pca.fit(combined)

    mri_pca = pca.transform(mri_flat)
    us_pca = pca.transform(us_flat)
    var_exp = float(np.sum(pca.explained_variance_ratio_))

    # 4. Joint min-max normalization to [0, 255]
    all_pca = np.vstack([mri_pca, us_pca])
    min_val = all_pca.min(axis=0)
    max_val = all_pca.max(axis=0)
    denom = np.where((max_val - min_val) < 1e-6, 1e-6, max_val - min_val)

    mri_norm = np.clip((mri_pca - min_val) / denom * 255.0, 0, 255).astype(np.uint8)
    us_norm = np.clip((us_pca - min_val) / denom * 255.0, 0, 255).astype(np.uint8)

    rgb_mri = mri_norm.reshape(h, w, 3)
    rgb_us = us_norm.reshape(h, w, 3)

    return rgb_mri, rgb_us, var_exp


def render_latent_rgb_visual(
    data_path: Path = Path("data/resect_base/RESECT/NIFTI/Case1/MRI/Case1-FLAIR.nii.gz"),
    checkpoint_path: Path = Path("checkpoints/best_model_pose_pipeline.pth"),
    output_path: Path = Path("figures/latent_feature_pca.png"),
    device_name: Optional[str] = None,
) -> None:
    """Generate side-by-side RGB latent space feature maps for paired MRI and US slices."""
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Generating Latent RGB Visualisation on device: {device}")

    # 1. Load volume
    from scripts.calibrate_patient import volume_options
    _cfg = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("config") if checkpoint_path.exists() else None
    vol_tensor, vol_shape, spacing = load_or_create_volume(data_path, device, **volume_options(_cfg))

    # 2. Extract central axial slice (z=0, identity rotation)
    # Same reslicer geometry and simulator as the checkpoint was trained with
    from scripts.calibrate_patient import pipeline_from_config
    from src.utils.metrics import volume_half_extent_mm
    cfg = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("config") if checkpoint_path.exists() else None
    reslicer, simulator, *_ = pipeline_from_config(cfg, device)

    # Standard canonical central slice pose: [0, 0, 0, 1, 0, 0, 0, 1, 0]
    p_center = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
    mri_slice = reslicer(vol_tensor, p_center, half_extent_mm=volume_half_extent_mm(vol_shape, spacing))
    us_slice = simulator(mri_slice)

    # 3. Instantiate encoders and load weights
    nn1 = NN1(embedding_dim=128).to(device)
    nn2 = NN2(embedding_dim=128).to(device)

    if checkpoint_path.exists():
        print(f"[*] Ingesting model weights from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "nn1_state_dict" in ckpt:
                nn1.load_state_dict(ckpt["nn1_state_dict"])
            if "nn2_state_dict" in ckpt:
                nn2.load_state_dict(ckpt["nn2_state_dict"])
            print("    Successfully loaded NN1 and NN2 weights.")
        except Exception as e:
            print(f"[!] Notice: Could not load checkpoint ({e}). Using initialized weights.")
    else:
        print(f"[*] Notice: Checkpoint {checkpoint_path} not found. Using initialized weights.")

    nn1.eval()
    nn2.eval()

    # 4. Extract global embeddings and spatial feature maps
    with torch.no_grad():
        emb1 = nn1(mri_slice)  # [1, 128]
        emb2 = nn2(us_slice)   # [1, 128]
        feat1 = nn1.extract_spatial_features(mri_slice)  # [1, 512, 8, 8]
        feat2 = nn2.extract_spatial_features(us_slice)   # [1, 512, 8, 8]

    # Global latent similarity metrics
    cos_sim = float(F.cosine_similarity(emb1, emb2, dim=-1).item())
    l2_dist = float(torch.norm(emb1 - emb2, p=2, dim=-1).item())
    print(f"[*] Global Embedding Metrics: Cosine Similarity = {cos_sim:.4f} | L2 Distance = {l2_dist:.4f}")

    # 5. Project spatial feature maps to RGB via PCA
    rgb_mri, rgb_us, var_exp = project_features_to_rgb(feat1, feat2, target_size=(227, 227))
    print(f"[*] Channel-wise PCA complete. Top-3 components explain {var_exp * 100:.1f}% of spatial variance.")

    # 6. Render publication-quality 2x2 comparison figure
    plt.style.use("seaborn-v0_8-white" if "seaborn-v0_8-white" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # Top-Left: Original MRI
    axes[0, 0].imshow(mri_slice[0, 0].detach().cpu().numpy(), cmap="gray", origin="lower")
    axes[0, 0].set_title("A) Clean Preoperative MRI Slice", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Top-Right: Synthetic US
    axes[0, 1].imshow(us_slice[0, 0].detach().cpu().numpy(), cmap="gray", origin="lower")
    axes[0, 1].set_title("B) Synthetic Intraoperative Ultrasound Slice", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Bottom-Left: MRI Latent RGB
    axes[1, 0].imshow(rgb_mri, origin="lower")
    axes[1, 0].set_title("C) MRI Latent Space Spatial Features (RGB)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Bottom-Right: US Latent RGB
    axes[1, 1].imshow(rgb_us, origin="lower")
    axes[1, 1].set_title("D) US Latent Space Spatial Features (RGB)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Latent RGB feature map visual successfully saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Latent Space RGB Feature Map Visual")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/resect_base/RESECT/NIFTI/Case1/MRI/Case1-FLAIR.nii.gz"),
        help="Path to Case1 preoperative T2-FLAIR MRI volume",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model_pose_pipeline.pth"),
        help="Path to trained pipeline checkpoint",
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=Path("figures/latent_feature_pca.png"),
        help="Path to export RGB visualizer figure",
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()

    render_latent_rgb_visual(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        output_path=args.output_plot,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
