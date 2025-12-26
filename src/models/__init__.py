"""
VISTA-Embed Model Components

Contains the VistaLightningModule and associated loss functions
for connectomics instance segmentation.

Uses einops for tensor operations.
"""

from .vista_wrapper import (
    VistaLightningModule,
    VistaEmbedSmall,
    VistaEmbedLarge,
    EmbeddingNormalizer,
    ConvBlock3D
)
from .components.discriminative import (
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
    CombinedInstanceLoss,
    ContrastiveLoss,
    AffinityLoss
)
from .metrics import (
    InstanceSegmentationMetric,
    InstanceSegmentationMetrics,
    compute_adjusted_rand_score,
    compute_rand_score,
    compute_normalized_mutual_info,
    compute_variation_of_information,
    cluster_embeddings_for_metrics
)

__all__ = [
    # Models
    'VistaLightningModule',
    'VistaEmbedSmall',
    'VistaEmbedLarge',
    # Components
    'EmbeddingNormalizer',
    'ConvBlock3D',
    # Losses
    'DiscriminativeLoss',
    'DiscriminativeLossVectorized',
    'CombinedInstanceLoss',
    'ContrastiveLoss',
    'AffinityLoss',
    # Metrics (TorchMetrics-based)
    'InstanceSegmentationMetric',
    'InstanceSegmentationMetrics',
    'compute_adjusted_rand_score',
    'compute_rand_score',
    'compute_normalized_mutual_info',
    'compute_variation_of_information',
    'cluster_embeddings_for_metrics'
]
