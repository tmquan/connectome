"""
Discriminative Loss for Instance Segmentation

Implementation based on:
"Semantic Instance Segmentation with a Discriminative Loss Function"
De Brabandere et al., 2017 (arXiv:1708.02551)

Uses einops for clean tensor operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from einops import rearrange, repeat, reduce, einsum


class DiscriminativeLoss(nn.Module):
    """
    Discriminative Loss for metric learning in instance segmentation.
    
    Transforms dense voxel predictions into embeddings that cluster
    by instance, enabling post-hoc separation via clustering.
    
    Loss components:
    1. Variance Loss (L_var): Pull embeddings towards instance mean
    2. Distance Loss (L_dist): Push instance means apart
    3. Regularization Loss (L_reg): Keep means near origin
    
    Args:
        delta_var: Margin for variance loss (default: 0.5)
        delta_dist: Margin for distance loss (default: 1.5)
        norm: Norm degree for distance (default: 2)
        alpha: Weight for variance term (default: 1.0)
        beta: Weight for distance term (default: 1.0)
        gamma: Weight for regularization term (default: 0.001)
        ignore_label: Label to ignore (default: 0)
    """
    
    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        norm: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        ignore_label: int = 0
    ):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_label = ignore_label
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor,
        semantic_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute discriminative loss.
        
        Args:
            embedding: Dense embeddings (B, E, D, H, W)
            instance_mask: Instance IDs (B, 1, D, H, W)
            semantic_mask: Optional semantic mask for region restriction
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Individual loss components
        """
        batch_size = embedding.shape[0]
        device = embedding.device
        
        total_var_loss = torch.tensor(0.0, device=device)
        total_dist_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for b in range(batch_size):
            # Extract single sample using einops
            # (E, D, H, W) -> (E, N) where N = D*H*W
            emb = rearrange(embedding[b], 'e d h w -> e (d h w)')
            inst = rearrange(instance_mask[b, 0], 'd h w -> (d h w)')
            
            # Apply semantic mask if provided
            if semantic_mask is not None:
                sem = rearrange(semantic_mask[b, 0], 'd h w -> (d h w)')
                valid_mask = sem > 0
            else:
                valid_mask = inst != self.ignore_label
            
            # Get unique instances
            unique_instances = torch.unique(inst[valid_mask])
            unique_instances = unique_instances[unique_instances != self.ignore_label]
            num_instances = len(unique_instances)
            
            if num_instances < 1:
                continue
            
            valid_batches += 1
            
            # Compute cluster means using einops
            cluster_means, cluster_sizes = self._compute_cluster_means(
                emb, inst, unique_instances
            )
            
            if len(cluster_means) < 1:
                continue
            
            # Stack means: (C, E) where C is number of clusters
            cluster_means = torch.stack(cluster_means)
            num_clusters = len(cluster_means)
            
            # 1. Variance Loss
            var_loss = self._variance_loss(emb, inst, unique_instances, cluster_means)
            total_var_loss = total_var_loss + var_loss
            
            # 2. Distance Loss
            if num_clusters > 1:
                dist_loss = self._distance_loss(cluster_means)
                total_dist_loss = total_dist_loss + dist_loss
            
            # 3. Regularization Loss
            reg_loss = self._regularization_loss(cluster_means)
            total_reg_loss = total_reg_loss + reg_loss
        
        # Average over batches
        if valid_batches > 0:
            total_var_loss = total_var_loss / valid_batches
            total_dist_loss = total_dist_loss / valid_batches
            total_reg_loss = total_reg_loss / valid_batches
        
        # Weighted sum
        total_loss = (
            self.alpha * total_var_loss +
            self.beta * total_dist_loss +
            self.gamma * total_reg_loss
        )
        
        loss_dict = {
            'loss_var': total_var_loss,
            'loss_dist': total_dist_loss,
            'loss_reg': total_reg_loss,
            'loss_total': total_loss,
            'num_batches': valid_batches
        }
        
        return total_loss, loss_dict
    
    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute mean embedding for each instance cluster."""
        cluster_means = []
        cluster_sizes = []
        
        for instance_id in unique_instances:
            mask = inst == instance_id
            if mask.sum() == 0:
                continue
            
            # Select embeddings for this instance: (E, N_i)
            instance_emb = emb[:, mask]
            
            # Compute mean using einops reduce
            mean_emb = reduce(instance_emb, 'e n -> e', 'mean')
            
            cluster_means.append(mean_emb)
            cluster_sizes.append(mask.sum().float())
        
        return cluster_means, cluster_sizes
    
    def _variance_loss(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_means: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variance loss: pull embeddings to cluster means.
        
        L_var = (1/C) * Σ_c (1/N_c) * Σ_i [max(0, ||μ_c - x_i|| - δ_v)]²
        """
        var_loss = torch.tensor(0.0, device=emb.device)
        num_clusters = 0
        
        for idx, instance_id in enumerate(unique_instances):
            mask = inst == instance_id
            if mask.sum() == 0:
                continue
            
            # Instance embeddings: (E, N_i)
            instance_emb = emb[:, mask]
            
            # Mean embedding: (E,) -> (E, 1) for broadcasting
            mean_emb = rearrange(cluster_means[idx], 'e -> e 1')
            
            # Compute distances using einops
            # diff: (E, N_i)
            diff = instance_emb - mean_emb
            
            # L2 norm: (N_i,)
            distances = torch.norm(diff, p=self.norm, dim=0)
            
            # Hinge loss with margin
            hinged = F.relu(distances - self.delta_var) ** 2
            
            # Mean over instance voxels
            var_loss = var_loss + reduce(hinged, 'n -> ', 'mean')
            num_clusters += 1
        
        if num_clusters > 0:
            var_loss = var_loss / num_clusters
        
        return var_loss
    
    def _distance_loss(self, cluster_means: torch.Tensor) -> torch.Tensor:
        """
        Compute distance loss: push cluster means apart.
        
        L_dist = (1/C(C-1)) * Σ_{a≠b} [max(0, 2δ_d - ||μ_a - μ_b||)]²
        """
        num_clusters = cluster_means.shape[0]
        
        if num_clusters < 2:
            return torch.tensor(0.0, device=cluster_means.device)
        
        # Compute pairwise distances using einops
        # cluster_means: (C, E)
        # Expand for pairwise computation
        means_a = rearrange(cluster_means, 'c e -> c 1 e')
        means_b = rearrange(cluster_means, 'c e -> 1 c e')
        
        # Pairwise differences: (C, C, E)
        diff = means_a - means_b
        
        # Pairwise distances: (C, C)
        distances = torch.norm(diff, p=self.norm, dim=2)
        
        # Hinge loss with margin 2*delta_dist
        hinged = F.relu(2 * self.delta_dist - distances) ** 2
        
        # Mask diagonal (same cluster)
        mask = ~torch.eye(num_clusters, dtype=torch.bool, device=cluster_means.device)
        
        # Mean over valid pairs
        dist_loss = hinged[mask].mean()
        
        return dist_loss
    
    def _regularization_loss(self, cluster_means: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss: penalize mean norms.
        
        L_reg = (1/C) * Σ_c ||μ_c||
        """
        # cluster_means: (C, E)
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        return reduce(norms, 'c -> ', 'mean')


class DiscriminativeLossVectorized(DiscriminativeLoss):
    """
    Vectorized implementation using einops for better GPU efficiency.
    
    Avoids explicit Python loops where possible using scatter operations
    and einops rearrangements.
    """
    
    def _variance_loss(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_means: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized variance loss using scatter operations."""
        num_instances = len(unique_instances)
        
        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device)
        
        # Create instance-to-index mapping
        max_inst = int(inst.max().item()) + 1
        inst_to_idx = torch.full((max_inst,), -1, device=emb.device, dtype=torch.long)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_to_idx[inst_id.long()] = idx
        
        # Map each voxel to cluster index
        cluster_indices = inst_to_idx[inst.long()]  # (N,)
        valid_mask = cluster_indices >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=emb.device)
        
        # Get valid embeddings and indices
        valid_emb = emb[:, valid_mask]  # (E, N_valid)
        valid_idx = cluster_indices[valid_mask]  # (N_valid,)
        
        # Gather cluster means for each voxel
        # cluster_means: (C, E) -> index by valid_idx
        gathered_means = cluster_means[valid_idx]  # (N_valid, E)
        gathered_means = rearrange(gathered_means, 'n e -> e n')
        
        # Compute distances
        diff = valid_emb - gathered_means  # (E, N_valid)
        distances = torch.norm(diff, p=self.norm, dim=0)  # (N_valid,)
        
        # Hinge loss
        hinged = F.relu(distances - self.delta_var) ** 2
        
        # Compute per-cluster mean using scatter
        cluster_losses = torch.zeros(num_instances, device=emb.device)
        cluster_counts = torch.zeros(num_instances, device=emb.device)
        
        cluster_losses.scatter_add_(0, valid_idx, hinged)
        cluster_counts.scatter_add_(0, valid_idx, torch.ones_like(hinged))
        
        # Avoid division by zero
        cluster_counts = torch.clamp(cluster_counts, min=1)
        
        # Mean per cluster, then mean across clusters
        per_cluster_loss = cluster_losses / cluster_counts
        var_loss = reduce(per_cluster_loss, 'c -> ', 'mean')
        
        return var_loss


class CombinedInstanceLoss(nn.Module):
    """
    Combined loss for instance segmentation with optional boundary loss.
    
    Uses einops for tensor operations.
    """
    
    def __init__(
        self,
        discriminative_config: Dict,
        use_boundary_loss: bool = False,
        boundary_weight: float = 0.5
    ):
        super().__init__()
        self.disc_loss = DiscriminativeLoss(**discriminative_config)
        self.use_boundary_loss = use_boundary_loss
        self.boundary_weight = boundary_weight
        
        if use_boundary_loss:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor,
        boundary_pred: Optional[torch.Tensor] = None,
        boundary_target: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss."""
        disc_loss, disc_dict = self.disc_loss(embedding, instance_mask, semantic_mask)
        
        total_loss = disc_loss
        loss_dict = disc_dict.copy()
        
        if self.use_boundary_loss and boundary_pred is not None and boundary_target is not None:
            boundary_loss = self.bce_loss(boundary_pred, boundary_target.float())
            total_loss = total_loss + self.boundary_weight * boundary_loss
            loss_dict['loss_boundary'] = boundary_loss
        
        loss_dict['loss_combined'] = total_loss
        
        return total_loss, loss_dict


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for embedding learning.
    
    Alternative to discriminative loss using pairwise comparisons
    with einops for efficient batch operations.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        num_samples: int = 1000
    ):
        super().__init__()
        self.margin = margin
        self.num_samples = num_samples
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive loss via sampled pairs.
        
        Args:
            embedding: (B, E, D, H, W)
            instance_mask: (B, 1, D, H, W)
        """
        batch_size = embedding.shape[0]
        device = embedding.device
        
        total_pos_loss = torch.tensor(0.0, device=device)
        total_neg_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for b in range(batch_size):
            # Flatten spatial dims
            emb = rearrange(embedding[b], 'e d h w -> (d h w) e')  # (N, E)
            inst = rearrange(instance_mask[b, 0], 'd h w -> (d h w)')  # (N,)
            
            # Sample pairs
            n_voxels = emb.shape[0]
            if n_voxels < 2:
                continue
            
            # Random pair indices
            idx1 = torch.randint(0, n_voxels, (self.num_samples,), device=device)
            idx2 = torch.randint(0, n_voxels, (self.num_samples,), device=device)
            
            # Get embeddings and labels
            emb1 = emb[idx1]  # (S, E)
            emb2 = emb[idx2]  # (S, E)
            label1 = inst[idx1]
            label2 = inst[idx2]
            
            # Same instance (positive) or different (negative)
            same_instance = (label1 == label2) & (label1 != 0)
            diff_instance = (label1 != label2) & (label1 != 0) & (label2 != 0)
            
            # Pairwise distances
            diff = emb1 - emb2
            distances = torch.norm(diff, dim=1)
            
            # Positive loss: pull together
            if same_instance.any():
                pos_loss = reduce(distances[same_instance] ** 2, 'n -> ', 'mean')
                total_pos_loss = total_pos_loss + pos_loss
            
            # Negative loss: push apart with margin
            if diff_instance.any():
                neg_loss = reduce(
                    F.relu(self.margin - distances[diff_instance]) ** 2,
                    'n -> ', 'mean'
                )
                total_neg_loss = total_neg_loss + neg_loss
            
            valid_batches += 1
        
        if valid_batches > 0:
            total_pos_loss = total_pos_loss / valid_batches
            total_neg_loss = total_neg_loss / valid_batches
        
        total_loss = total_pos_loss + total_neg_loss
        
        return total_loss, {
            'loss_pos': total_pos_loss,
            'loss_neg': total_neg_loss,
            'loss_total': total_loss
        }


class AffinityLoss(nn.Module):
    """
    Affinity-based loss for boundary prediction.
    
    Predicts local affinities (edge probabilities) between neighboring voxels.
    Uses einops for efficient neighbor extraction.
    """
    
    def __init__(self, offsets: Optional[List[Tuple[int, int, int]]] = None):
        super().__init__()
        # Default 6-connected neighborhood
        self.offsets = offsets or [
            (1, 0, 0), (0, 1, 0), (0, 0, 1)  # 3 principal directions
        ]
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        affinity_pred: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute affinity loss.
        
        Args:
            affinity_pred: Predicted affinities (B, 3, D, H, W)
            instance_mask: Instance labels (B, 1, D, H, W)
        """
        device = affinity_pred.device
        
        # Compute ground truth affinities
        affinity_gt = self._compute_affinity_targets(instance_mask)
        
        # Binary cross-entropy
        loss = self.bce_loss(affinity_pred, affinity_gt.float())
        
        return loss
    
    def _compute_affinity_targets(
        self,
        instance_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute affinity targets from instance mask."""
        # instance_mask: (B, 1, D, H, W)
        inst = instance_mask[:, 0]  # (B, D, H, W)
        
        affinities = []
        
        for offset in self.offsets:
            # Shift instance mask by offset
            shifted = self._shift_tensor(inst, offset)
            
            # Affinity = 1 if same instance (and both > 0)
            same_inst = (inst == shifted) & (inst > 0)
            affinities.append(same_inst.float())
        
        # Stack: (B, 3, D, H, W)
        return torch.stack(affinities, dim=1)
    
    def _shift_tensor(
        self,
        tensor: torch.Tensor,
        offset: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Shift tensor by offset with zero padding."""
        d, h, w = offset
        result = torch.zeros_like(tensor)
        
        # Compute valid ranges
        if d >= 0:
            src_d, dst_d = slice(None, -d if d else None), slice(d, None)
        else:
            src_d, dst_d = slice(-d, None), slice(None, d)
        
        if h >= 0:
            src_h, dst_h = slice(None, -h if h else None), slice(h, None)
        else:
            src_h, dst_h = slice(-h, None), slice(None, h)
        
        if w >= 0:
            src_w, dst_w = slice(None, -w if w else None), slice(w, None)
        else:
            src_w, dst_w = slice(-w, None), slice(None, w)
        
        result[:, dst_d, dst_h, dst_w] = tensor[:, src_d, src_h, src_w]
        
        return result
