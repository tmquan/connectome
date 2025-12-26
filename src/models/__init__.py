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
    'AffinityLoss'
]
