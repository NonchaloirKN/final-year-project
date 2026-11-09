"""
Supervised Multi-Positive InfoNCE Loss for Cross-Modal Registration.

Part of FCN05: Real-Time 2D Ultrasound to 3D MRI Slice Matching (EEE4022S).
Implements spatial-continuity anchored contrastive loss where physical Euclidean
distance <= delta (1.5 mm) defines multiple positive anchors in a batch to decouple
anatomical structural learning from patient-specific identity memorization.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPositiveInfoNCELoss(nn.Module):
    """
    Supervised Multi-Positive InfoNCE Contrastive Loss.

    Parameters
    ----------
    temperature : float, optional
        Softmax temperature scaling hyperparameter tau, by default 0.07.
    distance_threshold : float, optional
        Maximum physical Euclidean distance delta (in mm) between slice coordinates
        to be considered positive anchors, by default 1.5 mm.
    eps : float, optional
        Numerical stability constant, by default 1e-8.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        distance_threshold: float = 1.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.tau = float(temperature)
        self.delta = float(distance_threshold)
        self.eps = float(eps)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        features_mr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-positive InfoNCE loss over a batch of features and spatial coordinates.

        Supports two input signatures:
        1. Single combined feature tensor: forward(features, coordinates)
           - features: [Batch, D]
           - coordinates: [Batch, 3]
        2. Paired cross-modal tensors: forward(features_us, coordinates, features_mr)
           - features_us: [Batch, D]
           - coordinates: [Batch, 3] (shared physical locations)
           - features_mr: [Batch, D]

        Parameters
        ----------
        features : torch.Tensor
            Embeddings or first modality features [Batch, D].
        coordinates : torch.Tensor
            Physical 3D coordinates (X, Y, Z) in mm [Batch, 3].
        features_mr : Optional[torch.Tensor], optional
            Second modality embeddings if passing paired cross-modal tensors.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if features_mr is not None:
            # Concatenate multimodal features and duplicate coordinates
            # to form a joint 2*Batch space where corresponding cross-modal
            # slices share identical physical locations (distance = 0 <= delta)
            all_features = torch.cat([features, features_mr], dim=0)
            all_coordinates = torch.cat([coordinates, coordinates], dim=0)
        else:
            all_features = features
            all_coordinates = coordinates

        batch_size = all_features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=all_features.device, requires_grad=True)

        # 1. L2 Normalize feature projections
        feats_norm = F.normalize(all_features, p=2, dim=1)

        # 2. Pairwise cosine similarity matrix scaled by temperature tau
        sim_matrix = torch.matmul(feats_norm, feats_norm.T) / self.tau

        # 3. Pairwise physical Euclidean distance matrix
        dist_matrix = torch.cdist(all_coordinates, all_coordinates, p=2)

        # 4. Construct diagonal mask to exclude self-comparisons
        diag_mask = torch.eye(batch_size, device=all_features.device, dtype=torch.bool)

        # 5. Construct positive mask: physical distance <= delta, excluding self
        pos_mask = (dist_matrix <= self.delta) & (~diag_mask)

        # 6. Compute log-sum-exp denominator over all elements excluding self
        sim_matrix_masked = sim_matrix.masked_fill(diag_mask, -1e9)
        log_denom = torch.logsumexp(sim_matrix_masked, dim=1, keepdim=True)

        # 7. Log-probabilities: log(P_ij) = (z_i . z_j / tau) - log(sum_k!=i exp(z_i . z_k / tau))
        log_prob = sim_matrix - log_denom

        # 8. Average log-probability over positive anchors for each anchor
        num_pos = pos_mask.sum(dim=1)
        valid_anchors = num_pos > 0

        if not valid_anchors.any():
            # If no pair falls within distance threshold delta, return zero loss
            return torch.tensor(0.0, device=all_features.device, requires_grad=True)

        # Compute mean log probability over positive anchors for valid anchor slices
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)[valid_anchors] / num_pos[valid_anchors]

        # Total multi-positive loss is the negative mean across the batch
        loss = -pos_log_prob.mean()
        return loss
