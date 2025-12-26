"""
Instance Segmentation Metrics

Implements metrics for evaluating instance segmentation quality:
- Adjusted Rand Score (ARI) - using sklearn for correctness
- Normalized Mutual Information (NMI)
- Rand Score
- Variation of Information (VI)

Uses sklearn for ARI (guaranteed correct implementation).
Uses TorchMetrics for other clustering metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from torchmetrics import Metric, MetricCollection
from torchmetrics.clustering import (
    AdjustedRandScore,
    NormalizedMutualInfoScore,
    RandScore,
    AdjustedMutualInfoScore
)
from sklearn.metrics import adjusted_rand_score as sklearn_ari


def cluster_embeddings_for_metrics(
    embeddings: torch.Tensor,
    semantic_mask: torch.Tensor,
    bandwidth: float = 1.5,
    min_cluster_size: int = 50,
    use_euclidean: bool = True
) -> torch.Tensor:
    """
    Cluster embeddings to produce instance predictions for metric computation.
    
    Uses Euclidean distance-based greedy mean-shift-like clustering.
    This is more appropriate for discriminative loss embeddings which are
    trained to push different instances apart in Euclidean space.
    
    Args:
        embeddings: (E, D, H, W) or (E, H, W) embedding tensor
        semantic_mask: (D, H, W) or (H, W) binary foreground mask
        bandwidth: Distance threshold for clustering
                   For Euclidean: typical range 0.5-2.0 (depends on embedding scale)
                   For Cosine: typical range 0.1-0.3
        min_cluster_size: Minimum pixels per instance
        use_euclidean: Use Euclidean distance (True) or cosine distance (False)
        
    Returns:
        Instance segmentation tensor with same spatial shape as semantic_mask
    """
    device = embeddings.device
    spatial_shape = semantic_mask.shape
    
    # Flatten spatial dimensions
    if embeddings.ndim == 4:
        E, D, H, W = embeddings.shape
        emb_flat = embeddings.view(E, -1).T  # (D*H*W, E)
        mask_flat = semantic_mask.view(-1)  # (D*H*W,)
    else:
        E, H, W = embeddings.shape
        emb_flat = embeddings.view(E, -1).T  # (H*W, E)
        mask_flat = semantic_mask.view(-1)  # (H*W,)
    
    # Get foreground pixels
    fg_indices = torch.where(mask_flat > 0)[0]
    
    if len(fg_indices) == 0:
        return torch.zeros(spatial_shape, device=device, dtype=torch.long)
    
    fg_embeddings = emb_flat[fg_indices]  # (N, E)
    
    if not use_euclidean:
        # Normalize for cosine distance
        fg_embeddings = F.normalize(fg_embeddings, dim=1)
    
    # Greedy clustering
    N = fg_embeddings.shape[0]
    labels = torch.zeros(N, device=device, dtype=torch.long)
    current_label = 1
    
    # Subsample seeds for efficiency
    max_seeds = min(200, N)
    if N > max_seeds:
        seed_indices = torch.randperm(N, device=device)[:max_seeds]
    else:
        seed_indices = torch.arange(N, device=device)
    
    for seed_idx in seed_indices:
        if labels[seed_idx] > 0:
            continue
        
        seed_emb = fg_embeddings[seed_idx:seed_idx+1]  # (1, E)
        
        if use_euclidean:
            # Euclidean distance - more discriminative for untrained embeddings
            diff = fg_embeddings - seed_emb  # (N, E)
            distances = torch.norm(diff, dim=1)  # (N,)
        else:
            # Cosine distance
            similarities = torch.mm(fg_embeddings, seed_emb.T).squeeze()  # (N,)
            distances = 1 - similarities
        
        cluster_mask = (distances < bandwidth) & (labels == 0)
        
        if cluster_mask.sum() >= min_cluster_size:
            labels[cluster_mask] = current_label
            current_label += 1
    
    # Assign remaining unlabeled points to nearest cluster (if any clusters exist)
    if current_label > 1:
        unlabeled = labels == 0
        if unlabeled.any():
            # For each unlabeled point, find nearest labeled point
            labeled_mask = labels > 0
            labeled_emb = fg_embeddings[labeled_mask]
            labeled_labels = labels[labeled_mask]
            
            unlabeled_emb = fg_embeddings[unlabeled]
            
            # Compute distances to all labeled points (chunked for memory)
            chunk_size = 1000
            for i in range(0, unlabeled_emb.shape[0], chunk_size):
                chunk = unlabeled_emb[i:i+chunk_size]
                if use_euclidean:
                    dists = torch.cdist(chunk, labeled_emb)  # (chunk, labeled)
                else:
                    dists = 1 - torch.mm(chunk, labeled_emb.T)
                
                nearest_idx = dists.argmin(dim=1)
                labels[unlabeled.nonzero(as_tuple=True)[0][i:i+chunk_size]] = labeled_labels[nearest_idx]
    
    # Build output
    instance_map = torch.zeros(mask_flat.shape[0], device=device, dtype=torch.long)
    instance_map[fg_indices] = labels
    
    return instance_map.view(spatial_shape)


def compute_adjusted_rand_score(
    pred_instances: torch.Tensor,
    true_instances: torch.Tensor,
    semantic_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Adjusted Rand Score using sklearn (gold standard implementation).
    
    ARI is mathematically guaranteed to be in [-1, 1]:
    - 1.0: Perfect clustering
    - 0.0: Random clustering
    - <0: Worse than random
    
    Args:
        pred_instances: Predicted instance labels (D, H, W) or (H, W)
        true_instances: Ground truth instance labels, same shape
        semantic_mask: Optional mask to restrict evaluation to foreground
        
    Returns:
        ARI score as tensor in [-1, 1]
    """
    device = pred_instances.device
    
    # Move to CPU and convert to numpy for sklearn
    pred_flat = pred_instances.flatten().cpu().numpy()
    true_flat = true_instances.flatten().cpu().numpy()
    
    if semantic_mask is not None:
        mask_flat = semantic_mask.flatten().cpu().numpy() > 0
        pred_flat = pred_flat[mask_flat]
        true_flat = true_flat[mask_flat]
    
    # Edge case: no samples or too few
    if len(pred_flat) < 2:
        return torch.tensor(0.0, device=device)
    
    # Edge case: all same label in pred or true (degenerate clustering)
    n_pred_unique = len(np.unique(pred_flat))
    n_true_unique = len(np.unique(true_flat))
    
    if n_pred_unique == 1 or n_true_unique == 1:
        # ARI is undefined when one clustering has only one cluster
        return torch.tensor(0.0, device=device)
    
    # Use sklearn - guaranteed correct implementation
    ari = sklearn_ari(true_flat, pred_flat)
    
    return torch.tensor(ari, device=device, dtype=torch.float32)


def compute_rand_score(
    pred_instances: torch.Tensor,
    true_instances: torch.Tensor,
    semantic_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Rand Score (unadjusted) using TorchMetrics functional API.
    
    Rand Score ranges from 0 to 1:
    - 1.0: Perfect clustering
    - 0.5: Random clustering
    """
    from torchmetrics.functional.clustering import rand_score as tm_rand_score
    
    pred_flat = pred_instances.flatten()
    true_flat = true_instances.flatten()
    
    if semantic_mask is not None:
        mask_flat = semantic_mask.flatten() > 0
        pred_flat = pred_flat[mask_flat]
        true_flat = true_flat[mask_flat]
    
    if len(pred_flat) == 0:
        return torch.tensor(0.0, device=pred_instances.device)
    
    return tm_rand_score(pred_flat.long(), true_flat.long())


def compute_normalized_mutual_info(
    pred_instances: torch.Tensor,
    true_instances: torch.Tensor,
    semantic_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Normalized Mutual Information Score using TorchMetrics.
    
    NMI ranges from 0 to 1:
    - 1.0: Perfect clustering
    - 0.0: Independent clustering
    """
    from torchmetrics.functional.clustering import normalized_mutual_info_score as tm_nmi
    
    pred_flat = pred_instances.flatten()
    true_flat = true_instances.flatten()
    
    if semantic_mask is not None:
        mask_flat = semantic_mask.flatten() > 0
        pred_flat = pred_flat[mask_flat]
        true_flat = true_flat[mask_flat]
    
    if len(pred_flat) == 0:
        return torch.tensor(0.0, device=pred_instances.device)
    
    return tm_nmi(pred_flat.long(), true_flat.long())


def compute_variation_of_information(
    pred_instances: torch.Tensor,
    true_instances: torch.Tensor,
    semantic_mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute Variation of Information (VI) split and merge errors.
    
    VI = VI_split + VI_merge
    - VI_split: Over-segmentation error (H(pred|gt))
    - VI_merge: Under-segmentation error (H(gt|pred))
    
    Lower is better (0 = perfect).
    """
    pred_flat = pred_instances.flatten()
    true_flat = true_instances.flatten()
    device = pred_instances.device
    
    if semantic_mask is not None:
        mask_flat = semantic_mask.flatten() > 0
        pred_flat = pred_flat[mask_flat]
        true_flat = true_flat[mask_flat]
    
    if len(pred_flat) == 0:
        return {
            'vi_split': torch.tensor(0.0, device=device),
            'vi_merge': torch.tensor(0.0, device=device),
            'vi_total': torch.tensor(0.0, device=device)
        }
    
    # Move to numpy for computation (VI not in torchmetrics)
    pred_np = pred_flat.cpu().numpy()
    true_np = true_flat.cpu().numpy()
    
    pred_ids = np.unique(pred_np)
    true_ids = np.unique(true_np)
    n = len(pred_np)
    
    # H(pred|gt) - split error
    vi_split = 0.0
    for true_id in true_ids:
        true_mask = (true_np == true_id)
        p_gt = true_mask.sum() / n
        if p_gt == 0:
            continue
        for pred_id in pred_ids:
            p_joint = ((true_np == true_id) & (pred_np == pred_id)).sum() / n
            if p_joint > 0:
                vi_split -= p_joint * np.log2(p_joint / p_gt)
    
    # H(gt|pred) - merge error
    vi_merge = 0.0
    for pred_id in pred_ids:
        pred_mask = (pred_np == pred_id)
        p_pred = pred_mask.sum() / n
        if p_pred == 0:
            continue
        for true_id in true_ids:
            p_joint = ((true_np == true_id) & (pred_np == pred_id)).sum() / n
            if p_joint > 0:
                vi_merge -= p_joint * np.log2(p_joint / p_pred)
    
    return {
        'vi_split': torch.tensor(vi_split, device=device),
        'vi_merge': torch.tensor(vi_merge, device=device),
        'vi_total': torch.tensor(vi_split + vi_merge, device=device)
    }


class InstanceSegmentationMetric(Metric):
    """
    TorchMetrics-compatible metric for instance segmentation.
    
    Computes ARI by clustering embeddings and comparing to ground truth.
    Integrates with PyTorch Lightning for automatic metric aggregation.
    
    Usage in LightningModule:
        self.val_ari = InstanceSegmentationMetric()
        
        def validation_step(self, batch, batch_idx):
            ...
            self.val_ari.update(embeddings, true_instances, semantic_mask)
            self.log('val/ari', self.val_ari, on_epoch=True)
    """
    
    # TorchMetrics requires these
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    
    def __init__(
        self,
        bandwidth: float = 1.5,  # Euclidean distance threshold
        min_cluster_size: int = 20,
        use_euclidean: bool = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.min_cluster_size = min_cluster_size
        self.use_euclidean = use_euclidean
        
        # State variables for accumulation
        self.add_state("ari_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(
        self,
        embeddings: torch.Tensor,
        true_instances: torch.Tensor,
        semantic_mask: torch.Tensor
    ) -> None:
        """
        Update metric state with predictions and targets.
        
        Args:
            embeddings: (B, E, D, H, W) or (B, E, H, W) embeddings
            true_instances: (B, 1, D, H, W) or (B, D, H, W) ground truth
            semantic_mask: (B, 1, D, H, W) or (B, D, H, W) foreground mask
        """
        B = embeddings.shape[0]
        
        # Handle channel dimension in labels
        if true_instances.ndim == 5:
            true_instances = true_instances[:, 0]  # (B, D, H, W)
        if semantic_mask.ndim == 5:
            semantic_mask = semantic_mask[:, 0]  # (B, D, H, W)
        
        for b in range(B):
            # Cluster embeddings using Euclidean distance
            pred_instances = cluster_embeddings_for_metrics(
                embeddings[b],
                semantic_mask[b],
                bandwidth=self.bandwidth,
                min_cluster_size=self.min_cluster_size,
                use_euclidean=self.use_euclidean
            )
            
            # Compute ARI using TorchMetrics - guaranteed to be in [-1, 1]
            ari = compute_adjusted_rand_score(
                pred_instances,
                true_instances[b],
                semantic_mask[b]
            )
            
            self.ari_sum += ari
            self.count += 1
    
    def compute(self) -> torch.Tensor:
        """Compute final metric value - average of ARIs, guaranteed to be in [-1, 1]."""
        if self.count == 0:
            return torch.tensor(0.0, device=self.ari_sum.device)
        
        # Average of values in [-1, 1] is mathematically in [-1, 1]
        ari = self.ari_sum / self.count
        return ari


class InstanceSegmentationMetrics(nn.Module):
    """
    Collection of instance segmentation metrics using TorchMetrics.
    
    Computes:
    - Adjusted Rand Score (ARI)
    - Normalized Mutual Information (NMI)  
    - Variation of Information (VI)
    
    Usage:
        metrics = InstanceSegmentationMetrics()
        for batch in dataloader:
            metrics.update(embeddings, true_instances, mask)
        results = metrics.compute()
        metrics.reset()
    """
    
    def __init__(
        self,
        bandwidth: float = 1.5,  # Euclidean distance threshold
        min_cluster_size: int = 20,
        use_euclidean: bool = True
    ):
        super().__init__()
        self.bandwidth = bandwidth
        self.min_cluster_size = min_cluster_size
        self.use_euclidean = use_euclidean
        
        # Accumulators
        self.ari_scores: List[float] = []
        self.nmi_scores: List[float] = []
        self.vi_splits: List[float] = []
        self.vi_merges: List[float] = []
    
    def reset(self):
        """Reset accumulated metrics."""
        self.ari_scores = []
        self.nmi_scores = []
        self.vi_splits = []
        self.vi_merges = []
    
    def update(
        self,
        embeddings: torch.Tensor,
        true_instances: torch.Tensor,
        semantic_mask: torch.Tensor
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            embeddings: (B, E, D, H, W) or (B, E, H, W) embeddings
            true_instances: (B, D, H, W) or (B, H, W) ground truth
            semantic_mask: (B, D, H, W) or (B, H, W) foreground mask
        """
        B = embeddings.shape[0]
        
        for b in range(B):
            # Cluster embeddings using Euclidean distance
            pred_instances = cluster_embeddings_for_metrics(
                embeddings[b],
                semantic_mask[b],
                bandwidth=self.bandwidth,
                min_cluster_size=self.min_cluster_size,
                use_euclidean=self.use_euclidean
            )
            
            # Compute metrics using TorchMetrics
            ari = compute_adjusted_rand_score(pred_instances, true_instances[b], semantic_mask[b])
            nmi = compute_normalized_mutual_info(pred_instances, true_instances[b], semantic_mask[b])
            vi = compute_variation_of_information(pred_instances, true_instances[b], semantic_mask[b])
            
            self.ari_scores.append(ari.item())
            self.nmi_scores.append(nmi.item())
            self.vi_splits.append(vi['vi_split'].item())
            self.vi_merges.append(vi['vi_merge'].item())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics - averages are mathematically in valid ranges."""
        if len(self.ari_scores) == 0:
            return {
                'ari': 0.0,
                'nmi': 0.0,
                'vi_split': 0.0,
                'vi_merge': 0.0,
                'vi_total': 0.0
            }
        
        # Average of values in [-1, 1] is in [-1, 1]
        # Average of values in [0, 1] is in [0, 1]
        return {
            'ari': float(np.mean(self.ari_scores)),
            'nmi': float(np.mean(self.nmi_scores)),
            'vi_split': float(np.mean(self.vi_splits)),
            'vi_merge': float(np.mean(self.vi_merges)),
            'vi_total': float(np.mean(self.vi_splits) + np.mean(self.vi_merges))
        }
