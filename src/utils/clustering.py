"""
Post-Processing Clustering Utilities

Transforms dense voxel embeddings into instance segmentation masks.
Uses einops for clean tensor/array operations.

Supported algorithms:
- Mean Shift: Mode-finding without specifying cluster count
- HDBSCAN: Hierarchical density-based clustering
- Watershed: Marker-based watershed from embedding distances
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union, List
from scipy import ndimage
from scipy.ndimage import label as connected_components

# Use einops for numpy operations where applicable
try:
    from einops import rearrange, reduce, repeat
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy if needed."""
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    return arr


def _rearrange_embeddings(
    embeddings: np.ndarray,
    to_channel_last: bool = True
) -> np.ndarray:
    """
    Rearrange embedding array between channel-first and channel-last.
    
    Args:
        embeddings: (E, D, H, W) or (D, H, W, E)
        to_channel_last: If True, convert to (D, H, W, E)
    """
    if embeddings.ndim != 4:
        raise ValueError(f"Expected 4D array, got {embeddings.ndim}D")
    
    if HAS_EINOPS:
        if to_channel_last and embeddings.shape[0] < embeddings.shape[-1]:
            # Already channel-last
            return embeddings
        elif to_channel_last:
            return rearrange(embeddings, 'e d h w -> d h w e')
        else:
            return rearrange(embeddings, 'd h w e -> e d h w')
    else:
        # Fallback without einops
        if to_channel_last and embeddings.shape[0] < embeddings.shape[-1]:
            return embeddings
        elif to_channel_last:
            return np.transpose(embeddings, (1, 2, 3, 0))
        else:
            return np.transpose(embeddings, (3, 0, 1, 2))


def cluster_embeddings(
    embeddings: Union[np.ndarray, torch.Tensor],
    foreground_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    method: str = 'mean_shift',
    **kwargs
) -> np.ndarray:
    """
    Main entry point for embedding-to-instance conversion.
    
    Args:
        embeddings: Dense embeddings (E, D, H, W) or (D, H, W, E)
        foreground_mask: Binary mask for valid regions (D, H, W)
        method: Clustering algorithm ('mean_shift', 'hdbscan', 'watershed')
        
    Returns:
        Instance mask (D, H, W)
    """
    embeddings = _to_numpy(embeddings)
    if foreground_mask is not None:
        foreground_mask = _to_numpy(foreground_mask).astype(bool)
    
    # Ensure channel-last format
    embeddings = _rearrange_embeddings(embeddings, to_channel_last=True)
    
    if foreground_mask is None:
        foreground_mask = np.ones(embeddings.shape[:3], dtype=bool)
    
    if method == 'mean_shift':
        return mean_shift_clustering(embeddings, foreground_mask, **kwargs)
    elif method == 'hdbscan':
        return hdbscan_clustering(embeddings, foreground_mask, **kwargs)
    elif method == 'watershed':
        return watershed_from_embeddings(embeddings, foreground_mask, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def mean_shift_clustering(
    embeddings: np.ndarray,
    foreground_mask: np.ndarray,
    bandwidth: float = 0.5,
    min_bin_freq: int = 100,
    n_jobs: int = -1,
    subsample_ratio: float = 0.1
) -> np.ndarray:
    """
    Mean Shift clustering.
    
    Args:
        embeddings: (D, H, W, E)
        foreground_mask: (D, H, W)
        bandwidth: Kernel bandwidth
        min_bin_freq: Minimum cluster size
        n_jobs: Parallel jobs
        subsample_ratio: Subsampling for efficiency
    """
    from sklearn.cluster import MeanShift
    
    spatial_shape = embeddings.shape[:3]
    
    # Extract foreground embeddings using einops-style masking
    fg_coords = np.where(foreground_mask)
    fg_embeddings = embeddings[fg_coords]  # (N, E)
    
    if len(fg_embeddings) == 0:
        return np.zeros(spatial_shape, dtype=np.int32)
    
    # Subsample for efficiency
    n_points = len(fg_embeddings)
    if n_points > 50000 and subsample_ratio < 1.0:
        subsample_idx = np.random.choice(
            n_points, int(n_points * subsample_ratio), replace=False
        )
        fit_embeddings = fg_embeddings[subsample_idx]
    else:
        fit_embeddings = fg_embeddings
    
    # Fit and predict
    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        min_bin_freq=min_bin_freq,
        n_jobs=n_jobs
    )
    ms.fit(fit_embeddings)
    labels = ms.predict(fg_embeddings)
    
    # Create output mask
    instance_mask = np.zeros(spatial_shape, dtype=np.int32)
    instance_mask[fg_coords] = labels + 1
    
    return remove_small_objects(instance_mask, min_size=min_bin_freq)


def hdbscan_clustering(
    embeddings: np.ndarray,
    foreground_mask: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.5,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    HDBSCAN clustering for varying density.
    
    Args:
        embeddings: (D, H, W, E)
        foreground_mask: (D, H, W)
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("Install hdbscan: pip install hdbscan")
    
    spatial_shape = embeddings.shape[:3]
    
    fg_coords = np.where(foreground_mask)
    fg_embeddings = embeddings[fg_coords]
    
    if len(fg_embeddings) == 0:
        return np.zeros(spatial_shape, dtype=np.int32)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(fg_embeddings)
    
    instance_mask = np.zeros(spatial_shape, dtype=np.int32)
    valid_labels = labels >= 0
    
    if np.any(valid_labels):
        instance_mask[
            fg_coords[0][valid_labels],
            fg_coords[1][valid_labels],
            fg_coords[2][valid_labels]
        ] = labels[valid_labels] + 1
    
    return instance_mask


def watershed_from_embeddings(
    embeddings: np.ndarray,
    foreground_mask: np.ndarray,
    seed_threshold: float = 0.3,
    min_seed_size: int = 50,
    connectivity: int = 1
) -> np.ndarray:
    """
    Watershed segmentation using embedding-based distance transform.
    """
    from skimage.segmentation import watershed
    
    spatial_shape = embeddings.shape[:3]
    
    if not np.any(foreground_mask):
        return np.zeros(spatial_shape, dtype=np.int32)
    
    # Compute distance field
    distance_field = compute_embedding_distance_field(embeddings, foreground_mask)
    
    # Find seeds
    seeds = find_embedding_seeds(
        distance_field, foreground_mask, seed_threshold, min_seed_size
    )
    
    markers, num_features = connected_components(seeds)
    
    if num_features == 0:
        return foreground_mask.astype(np.int32)
    
    return watershed(distance_field, markers, mask=foreground_mask, connectivity=connectivity)


def compute_embedding_distance_field(
    embeddings: np.ndarray,
    foreground_mask: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Compute distance field from embeddings using local variance.
    
    Uses einops-style operations for computing local statistics.
    """
    from scipy.ndimage import uniform_filter
    
    spatial_shape = embeddings.shape[:3]
    embedding_dim = embeddings.shape[3]
    
    # Compute local mean for each embedding dimension
    local_mean = np.zeros_like(embeddings)
    for e in range(embedding_dim):
        local_mean[..., e] = uniform_filter(
            embeddings[..., e], size=window_size, mode='reflect'
        )
    
    # Compute distance from local mean
    # Using einops-style reduction: sqrt(sum((emb - local_mean)^2, axis=-1))
    diff_squared = (embeddings - local_mean) ** 2
    if HAS_EINOPS:
        distance_field = np.sqrt(reduce(diff_squared, 'd h w e -> d h w', 'sum'))
    else:
        distance_field = np.sqrt(np.sum(diff_squared, axis=-1))
    
    distance_field[~foreground_mask] = np.inf
    
    return distance_field


def find_embedding_seeds(
    distance_field: np.ndarray,
    foreground_mask: np.ndarray,
    threshold: float = 0.3,
    min_size: int = 50
) -> np.ndarray:
    """Find seed regions for watershed."""
    seeds = (distance_field < threshold) & foreground_mask
    seeds = remove_small_objects(seeds.astype(np.int32), min_size=min_size)
    return seeds > 0


def remove_small_objects(
    label_mask: np.ndarray,
    min_size: int = 100
) -> np.ndarray:
    """Remove small connected components."""
    output = np.zeros_like(label_mask)
    unique_labels = np.unique(label_mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    new_label = 1
    for lab in unique_labels:
        binary = label_mask == lab
        labeled, num_features = connected_components(binary)
        
        for i in range(1, num_features + 1):
            component = labeled == i
            if component.sum() >= min_size:
                output[component] = new_label
                new_label += 1
    
    return output


def merge_overlapping_instances(
    instances: List[np.ndarray],
    coordinates: List[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int],
    overlap_threshold: float = 0.5
) -> np.ndarray:
    """
    Merge overlapping instance predictions from sliding window.
    
    Uses einops for efficient array operations.
    """
    output = np.zeros(output_shape, dtype=np.int32)
    label_mapping = {}
    current_max_label = 0
    
    for inst, (z, y, x) in zip(instances, coordinates):
        d, h, w = inst.shape
        
        # Extract existing labels in this region
        existing = output[z:z+d, y:y+h, x:x+w]
        
        for new_label in np.unique(inst):
            if new_label == 0:
                continue
            
            new_mask = inst == new_label
            
            # Find overlapping existing labels
            overlapping_labels = np.unique(existing[new_mask])
            overlapping_labels = overlapping_labels[overlapping_labels > 0]
            
            if len(overlapping_labels) == 0:
                # New instance
                current_max_label += 1
                output[z:z+d, y:y+h, x:x+w][new_mask] = current_max_label
            else:
                # Check overlap ratio
                best_label = overlapping_labels[0]
                best_overlap = 0
                
                for ol in overlapping_labels:
                    overlap = np.sum(new_mask & (existing == ol))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_label = ol
                
                # Merge if sufficient overlap
                overlap_ratio = best_overlap / new_mask.sum()
                if overlap_ratio >= overlap_threshold:
                    output[z:z+d, y:y+h, x:x+w][new_mask] = best_label
                else:
                    current_max_label += 1
                    output[z:z+d, y:y+h, x:x+w][new_mask] = current_max_label
    
    return output


def evaluate_clustering(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    metrics: List[str] = ['vi', 'rand', 'adapted_rand']
) -> Dict[str, float]:
    """
    Evaluate clustering quality.
    
    Args:
        prediction: Predicted instance mask
        ground_truth: Ground truth instance mask
        metrics: Metrics to compute
    """
    from skimage.metrics import variation_of_information, adapted_rand_error
    
    results = {}
    
    pred_flat = prediction.ravel()
    gt_flat = ground_truth.ravel()
    
    if 'vi' in metrics:
        vi_split, vi_merge = variation_of_information(gt_flat, pred_flat)
        results['vi_split'] = vi_split
        results['vi_merge'] = vi_merge
        results['vi_total'] = vi_split + vi_merge
    
    if 'adapted_rand' in metrics:
        error, precision, recall = adapted_rand_error(gt_flat, pred_flat)
        results['adapted_rand_error'] = error
        results['adapted_rand_precision'] = precision
        results['adapted_rand_recall'] = recall
    
    return results


class EmbeddingClusterer:
    """
    Stateful clustering class for batch processing.
    
    Maintains consistent parameters across multiple volumes.
    """
    
    def __init__(self, method: str = 'mean_shift', **method_kwargs):
        self.method = method
        self.method_kwargs = method_kwargs
    
    def __call__(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        foreground_mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> np.ndarray:
        """Cluster embeddings."""
        return cluster_embeddings(
            embeddings, foreground_mask, self.method, **self.method_kwargs
        )
    
    def process_batch(
        self,
        embeddings_batch: Union[np.ndarray, torch.Tensor],
        foreground_masks: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> List[np.ndarray]:
        """
        Process batch of volumes.
        
        Uses einops for batch indexing.
        """
        embeddings_batch = _to_numpy(embeddings_batch)
        batch_size = embeddings_batch.shape[0]
        
        results = []
        for i in range(batch_size):
            emb = embeddings_batch[i]
            mask = foreground_masks[i] if foreground_masks is not None else None
            results.append(self(emb, mask))
        
        return results


class EmbeddingVisualizer:
    """
    Visualize embeddings using dimensionality reduction.
    
    Useful for debugging and understanding embedding space.
    """
    
    def __init__(self, method: str = 'pca', n_components: int = 3):
        self.method = method
        self.n_components = n_components
        self._reducer = None
    
    def fit_transform(
        self,
        embeddings: np.ndarray,
        foreground_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reduce embedding dimensionality for visualization.
        
        Args:
            embeddings: (D, H, W, E) or (E, D, H, W)
            foreground_mask: Optional mask
            
        Returns:
            Reduced embeddings (D, H, W, n_components)
        """
        embeddings = _rearrange_embeddings(embeddings, to_channel_last=True)
        spatial_shape = embeddings.shape[:3]
        embedding_dim = embeddings.shape[3]
        
        # Flatten for reduction
        if HAS_EINOPS:
            flat_emb = rearrange(embeddings, 'd h w e -> (d h w) e')
        else:
            flat_emb = embeddings.reshape(-1, embedding_dim)
        
        # Apply dimensionality reduction
        if self.method == 'pca':
            from sklearn.decomposition import PCA
            self._reducer = PCA(n_components=self.n_components)
        elif self.method == 'umap':
            try:
                import umap
                self._reducer = umap.UMAP(n_components=self.n_components)
            except ImportError:
                raise ImportError("Install umap: pip install umap-learn")
        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            self._reducer = TSNE(n_components=self.n_components)
        
        reduced = self._reducer.fit_transform(flat_emb)
        
        # Reshape back
        if HAS_EINOPS:
            return rearrange(
                reduced,
                '(d h w) c -> d h w c',
                d=spatial_shape[0],
                h=spatial_shape[1],
                w=spatial_shape[2]
            )
        else:
            return reduced.reshape(*spatial_shape, self.n_components)
    
    def to_rgb(
        self,
        reduced_embeddings: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert reduced embeddings to RGB for visualization.
        
        Args:
            reduced_embeddings: (D, H, W, 3)
            normalize: Normalize to [0, 255]
        """
        if reduced_embeddings.shape[-1] != 3:
            raise ValueError("Need 3 components for RGB visualization")
        
        rgb = reduced_embeddings.copy()
        
        if normalize:
            for c in range(3):
                channel = rgb[..., c]
                vmin, vmax = channel.min(), channel.max()
                if vmax > vmin:
                    rgb[..., c] = (channel - vmin) / (vmax - vmin)
            rgb = (rgb * 255).astype(np.uint8)
        
        return rgb
