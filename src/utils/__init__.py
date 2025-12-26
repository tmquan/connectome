"""
Utility Functions

Contains label registry, clustering utilities, and helper functions.
Uses einops for tensor/array operations.
"""

from .registry import (
    LabelRegistry,
    LabelInfo,
    ConnectomicsLabels,
    create_embedding_init_strategy,
    validate_label_consistency
)
from .clustering import (
    cluster_embeddings,
    mean_shift_clustering,
    hdbscan_clustering,
    watershed_from_embeddings,
    compute_embedding_distance_field,
    remove_small_objects,
    merge_overlapping_instances,
    evaluate_clustering,
    EmbeddingClusterer,
    EmbeddingVisualizer
)

__all__ = [
    # Registry
    'LabelRegistry',
    'LabelInfo',
    'ConnectomicsLabels',
    'create_embedding_init_strategy',
    'validate_label_consistency',
    # Clustering
    'cluster_embeddings',
    'mean_shift_clustering',
    'hdbscan_clustering',
    'watershed_from_embeddings',
    'compute_embedding_distance_field',
    'remove_small_objects',
    'merge_overlapping_instances',
    'evaluate_clustering',
    'EmbeddingClusterer',
    'EmbeddingVisualizer'
]
