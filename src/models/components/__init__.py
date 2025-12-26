"""
Model Components

Custom loss functions and network components for VISTA-Embed.
Uses einops for tensor operations.
"""

from .discriminative import (
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
    CombinedInstanceLoss,
    ContrastiveLoss,
    AffinityLoss
)

__all__ = [
    'DiscriminativeLoss',
    'DiscriminativeLossVectorized',
    'CombinedInstanceLoss',
    'ContrastiveLoss',
    'AffinityLoss'
]
