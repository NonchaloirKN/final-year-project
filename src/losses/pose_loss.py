"""
MultiModalPoseLoss: Joint Modality and Continuous 6D Pose Regression Objective.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Formulates the composite loss function specified in Prof. Fred Nicolls' technical brief:
L_total = ||EMB1 - EMB2||_2^2 + alpha * (L_pose(NN3(EMB1), p) + L_pose(NN3(EMB2), p))

Loss Terms:
1. Modality Alignment Loss: Mean Squared Error (MSE) between EMB1 (MRI) and EMB2 (US) embeddings.
2. Translation Loss: MSE between predicted and ground-truth normalized translations [tx, ty, tz] in [-1, 1].
3. Rotation Loss: Chordal Frobenius Loss ||R_pred - R_gt||_F^2 between 3x3 SO(3) rotation matrices
   (Zhou continuous 6D Gram-Schmidt output vs ground truth), avoiding arccos gradient instabilities.
4. Optional cross-modal contrastive term (contrastive_weight > 0): symmetric in-batch InfoNCE over
   emb1 @ emb2^T / temperature plus mean (1 - cos) of matched pairs, which repels different slices
   and prevents the US embeddings collapsing into a narrow cone.

Translation scale: normalised grid translations are tiny (+/-20 mm is +/-0.157 on a 256 mm axis), so
trans_scale multiplies the translation residual before the MSE. Passing
volume_half_extent_mm(...) / 20 makes a 20 mm error on any axis equal 1.0, which puts the
translation and rotation terms on the same O(1) footing. The default of 1.0 keeps the legacy loss.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pose_regressor import rotation_6d_to_matrix
from src.utils.reslicer import axis_angle_to_matrix


TransScale = Union[float, Tuple[float, float, float], torch.Tensor]


def symmetric_info_nce(emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Symmetric in-batch cross-modal InfoNCE: matched pairs (i, i) are positives and every other
    slice in the batch is a negative, in both the MRI-to-US and US-to-MRI directions.

    Args:
        emb1: L2-normalised MRI embeddings [B, D].
        emb2: L2-normalised US embeddings [B, D].
        temperature: Softmax temperature tau.

    Returns:
        Scalar loss; ln(B) at chance, 0 for perfectly separated pairs.
    """
    logits = emb1 @ emb2.transpose(0, 1) / temperature
    targets = torch.arange(emb1.shape[0], device=emb1.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))


def chordal_frobenius_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chordal Frobenius rotation loss:
    L_rot = ||R_pred - R_gt||_F^2 = sum_{i,j} (R_pred_{ij} - R_gt_{ij})^2

    Properties:
    - Smooth and strictly convex near zero error.
    - Exactly 0.0 when R_pred == R_gt.
    - Equal to 8 * sin^2(theta / 2) where theta is the geodesic angle.
    - Completely avoids the arccos derivative singularity (1 / sqrt(1 - x^2)) at theta=0 and theta=pi.

    Args:
        R_pred: Predicted 3x3 rotation matrices of shape [B, 3, 3].
        R_gt: Ground truth 3x3 rotation matrices of shape [B, 3, 3].

    Returns:
        Scalar tensor with mean Frobenius squared distance.
    """
    diff = R_pred - R_gt
    return (diff ** 2).sum(dim=(-1, -2)).mean()


class MultiModalPoseLoss(nn.Module):
    """
    Joint Multi-Modal Metric Alignment and Continuous 6D Pose Regression Loss.

    L_total = modality_loss + alpha * (translation_loss + rotation_loss)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta_rot: float = 1.0,
        train_translation_only: bool = False,
        train_rotation_only: bool = False,
        trans_scale: TransScale = 1.0,
        contrastive_weight: float = 0.0,
        temperature: float = 0.1,
        axis_weights: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Args:
            alpha: Weight balancing modality alignment and geometric pose regression (default: 1.0).
            beta_rot: Relative weight for the rotation loss component within pose loss (default: 1.0).
            train_translation_only: If True, ignores rotation error and only backpropagates translation MSE.
            train_rotation_only: If True, ignores translation error and only backpropagates rotation loss.
            trans_scale: Scalar or per-axis factor applied to the translation residual before the MSE
                         (default 1.0 = legacy). Can be overridden per call in forward().
            contrastive_weight: Weight of the symmetric InfoNCE + (1 - cos) term added to the modality
                                loss (default 0.0 = legacy MSE alignment only).
            temperature: InfoNCE temperature tau (default: 0.1).
            axis_weights: Optional per-axis weights (x, y, z) on the squared translation residual, e.g.
                          (1, 1, 1.5) to emphasise elevation (default None = equal weights).
        """
        super().__init__()
        self.alpha = alpha
        self.beta_rot = beta_rot
        self.train_translation_only = train_translation_only
        self.train_rotation_only = train_rotation_only
        self.trans_scale = trans_scale
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.axis_weights = axis_weights
        self.last_metrics: Dict[str, float] = {}

    def _trans_mse(self, t_pred: torch.Tensor, t_gt: torch.Tensor, scale: TransScale) -> torch.Tensor:
        """MSE between translations after scaling the residual (scalar or per-axis) and weighting each axis."""
        s = torch.as_tensor(scale, dtype=t_pred.dtype, device=t_pred.device)
        sq = ((t_pred - t_gt) * s) ** 2
        if self.axis_weights is not None:
            sq = sq * torch.as_tensor(self.axis_weights, dtype=t_pred.dtype, device=t_pred.device)
        return sq.mean()

    def _parse_pose(
        self, pose: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse pose input into translation [B, 3] and rotation matrix [B, 3, 3].

        Args:
            pose: Tensor of shape [B, 9] (3 trans + 6 rot), [B, 6] (3 trans + 3 axis-angle),
                  or tuple (translation, rotation_matrix).

        Returns:
            translation: [B, 3]
            rotation_matrix: [B, 3, 3]
        """
        if isinstance(pose, (tuple, list)):
            return pose[0], pose[1]

        if not isinstance(pose, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or tuple, got {type(pose)}")

        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        n_dim = pose.shape[-1]
        trans = pose[..., :3]

        if n_dim == 9:
            # 3 translation + 6 continuous rotation
            R = rotation_6d_to_matrix(pose[..., 3:9])
        elif n_dim == 6:
            # 3 translation + 3 axis-angle rotation
            R = axis_angle_to_matrix(pose[..., 3:6])
        else:
            raise ValueError(f"Expected pose tensor with 6 or 9 columns, got {n_dim}")

        return trans, R

    def forward(
        self,
        emb1: Optional[torch.Tensor] = None,
        emb2: Optional[torch.Tensor] = None,
        p_pred: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        p_gt: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        p_pred_mri: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        train_translation_only: Optional[bool] = None,
        train_rotation_only: Optional[bool] = None,
        return_details: bool = False,
        trans_scale: Optional[TransScale] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute joint multimodal pose loss.

        Args:
            emb1: Optional normalized MRI embedding [B, 128].
            emb2: Optional normalized US embedding [B, 128].
            p_pred: Predicted pose (typically from NN3(emb2) on US branch or NN3(emb1) on Zero-US calibration),
                    shape [B, 9] or (t, R).
            p_gt: Ground truth pose, shape [B, 9], [B, 6], or (t, R).
            p_pred_mri: Optional predicted pose from NN3(emb1) on MRI branch.
            train_translation_only: Override instance flag to optimize only translation error.
            train_rotation_only: Override instance flag to optimize only rotation error.
            return_details: If True, returns (total_loss, metrics_dict).
            trans_scale: Optional per-call override of the translation residual scale (e.g. per volume).

        Returns:
            total_loss: Differentiable scalar tensor.
            metrics_dict (optional): Dictionary of individual loss components.
        """
        # 1. Parse ground truth pose
        if p_gt is None or p_pred is None:
            raise ValueError("p_gt and p_pred must be provided to MultiModalPoseLoss")
        t_gt, R_gt = self._parse_pose(p_gt)

        scale = self.trans_scale if trans_scale is None else trans_scale

        # 2. Modality alignment loss: MSE between normalized embeddings (if both provided),
        #    plus the optional in-batch contrastive term that repels non-matching slices
        contrastive_loss = torch.tensor(0.0, device=t_gt.device)
        if emb1 is not None and emb2 is not None:
            modality_loss = F.mse_loss(emb1, emb2)
            if self.contrastive_weight > 0.0 and emb1.shape[0] > 1:
                align = (1.0 - F.cosine_similarity(emb1, emb2, dim=-1)).mean()
                contrastive_loss = symmetric_info_nce(emb1, emb2, self.temperature) + align
                modality_loss = modality_loss + self.contrastive_weight * contrastive_loss
        else:
            modality_loss = torch.tensor(0.0, device=t_gt.device)

        # 3. Determine incremental training mode
        t_only = self.train_translation_only if train_translation_only is None else train_translation_only
        r_only = self.train_rotation_only if train_rotation_only is None else train_rotation_only

        # 4. Parse predicted US pose and compute pose loss components
        t_pred, R_pred = self._parse_pose(p_pred)
        trans_loss_us = self._trans_mse(t_pred, t_gt, scale)
        rot_loss_us = chordal_frobenius_loss(R_pred, R_gt)

        if t_only:
            pose_loss_us = trans_loss_us
        elif r_only:
            pose_loss_us = self.beta_rot * rot_loss_us
        else:
            pose_loss_us = trans_loss_us + self.beta_rot * rot_loss_us

        # 5. Optional MRI branch pose loss (full symmetric brief formulation)
        if p_pred_mri is not None:
            t_pred_mri, R_pred_mri = self._parse_pose(p_pred_mri)
            trans_loss_mri = self._trans_mse(t_pred_mri, t_gt, scale)
            rot_loss_mri = chordal_frobenius_loss(R_pred_mri, R_gt)

            if t_only:
                pose_loss_mri = trans_loss_mri
            elif r_only:
                pose_loss_mri = self.beta_rot * rot_loss_mri
            else:
                pose_loss_mri = trans_loss_mri + self.beta_rot * rot_loss_mri

            # Combined pose loss across both modalities
            trans_loss = 0.5 * (trans_loss_us + trans_loss_mri)
            rot_loss = 0.5 * (rot_loss_us + rot_loss_mri)
            pose_loss = 0.5 * (pose_loss_us + pose_loss_mri)
        else:
            trans_loss = trans_loss_us
            rot_loss = rot_loss_us
            pose_loss = pose_loss_us

        # 6. Composite total loss: L_total = modality_loss + alpha * pose_loss
        total_loss = modality_loss + self.alpha * pose_loss

        # Store metrics dictionary for logging
        self.last_metrics = {
            "total_loss": float(total_loss.item()),
            "modality_loss": float(modality_loss.item()),
            "contrastive_loss": float(contrastive_loss.item()),
            "translation_loss": float(trans_loss.item()),
            "rotation_loss": float(rot_loss.item()),
            "pose_loss": float(pose_loss.item()),
        }

        if return_details:
            return total_loss, self.last_metrics
        return total_loss
