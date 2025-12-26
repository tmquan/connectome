"""
VISTA-Embed Lightning Module

A PyTorch Lightning wrapper that adapts VISTA3D/SegResNet for connectomics
instance segmentation.

Uses einops for clean tensor operations throughout.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from .components.discriminative import DiscriminativeLoss, CombinedInstanceLoss


class ConvBlock3D(nn.Module):
    """3D Convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class VistaLightningModule(pl.LightningModule):
    """
    VISTA3D-inspired Lightning Module for Connectomics.
    
    Dual-head architecture:
    1. Semantic Head: Multi-class segmentation
    2. Instance Head: Dense embeddings for discriminative loss
    
    Uses einops for tensor manipulation.
    """
    
    def __init__(
        self,
        net_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        label_registry: Dict[str, int]
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.registry = label_registry
        self.net_config = net_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        
        # Class configuration
        self.instance_classes = ['neuron']
        self.semantic_classes = [
            k for k in label_registry.keys()
            if k not in self.instance_classes and k != 'background'
        ]
        
        # Initialize components
        self._init_backbone()
        self._init_heads()
        self._init_losses()
        self._init_class_embeddings()
        
        # Inference settings
        self.sw_batch_size = 4
        self.overlap = 0.5
    
    def _init_backbone(self):
        """Initialize SegResNet backbone."""
        self.backbone = SegResNet(
            spatial_dims=3,
            init_filters=self.net_config.get('init_filters', 32),
            in_channels=self.net_config.get('in_channels', 1),
            out_channels=self.net_config.get('feature_dim', 48),
            dropout_prob=self.net_config.get('dropout', 0.2),
            blocks_down=self.net_config.get('blocks_down', [1, 2, 2, 4]),
            blocks_up=self.net_config.get('blocks_up', [1, 1, 1])
        )
        
        pretrained_path = self.net_config.get('pretrained_path')
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
    
    def _init_heads(self):
        """Initialize segmentation heads."""
        feature_dim = self.net_config.get('feature_dim', 48)
        embedding_dim = self.net_config.get('embedding_dim', 16)
        num_semantic = len(self.semantic_classes) + 1
        
        # Semantic head
        self.semantic_head = nn.Sequential(
            ConvBlock3D(feature_dim, feature_dim // 2),
            nn.Conv3d(feature_dim // 2, num_semantic, kernel_size=1)
        )
        
        # Instance embedding head
        self.instance_head = nn.Sequential(
            ConvBlock3D(feature_dim, feature_dim // 2),
            nn.Conv3d(feature_dim // 2, embedding_dim, kernel_size=1)
        )
        
        # Optional boundary head
        self.use_boundary_head = self.net_config.get('use_boundary_head', False)
        if self.use_boundary_head:
            self.boundary_head = nn.Sequential(
                ConvBlock3D(feature_dim, feature_dim // 4),
                nn.Conv3d(feature_dim // 4, 3, kernel_size=1)
            )
    
    def _init_losses(self):
        """Initialize loss functions."""
        semantic_config = self.loss_config.get('semantic', {})
        self.sem_loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=semantic_config.get('dice_weight', 1.0),
            lambda_ce=semantic_config.get('ce_weight', 0.5)
        )
        
        disc_config = self.loss_config.get('discriminative', {})
        self.disc_loss_fn = DiscriminativeLoss(
            delta_var=disc_config.get('delta_var', 0.5),
            delta_dist=disc_config.get('delta_dist', 1.5),
            norm=disc_config.get('norm', 2),
            alpha=disc_config.get('alpha', 1.0),
            beta=disc_config.get('beta', 1.0),
            gamma=disc_config.get('gamma', 0.001)
        )
    
    def _init_class_embeddings(self):
        """Initialize learnable class embeddings (VISTA3D-style prompting)."""
        num_classes = len(self.registry)
        embedding_dim = self.net_config.get('class_embedding_dim', 64)
        
        self.class_embeddings = nn.Embedding(num_classes, embedding_dim)
        nn.init.normal_(self.class_embeddings.weight, mean=0.0, std=0.02)
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            model_dict = self.backbone.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained weights")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(
        self,
        x: torch.Tensor,
        class_prompt: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input (B, C, D, H, W)
            class_prompt: Optional class names for prompted segmentation
            
        Returns:
            Dictionary with 'semantic_logits', 'embeddings', and optionally 'boundary'
        """
        features = self.backbone(x)
        
        outputs = {
            'semantic_logits': self.semantic_head(features),
            'embeddings': self.instance_head(features)
        }
        
        if self.use_boundary_head:
            outputs['boundary'] = self.boundary_head(features)
        
        if class_prompt is not None:
            class_ids = torch.tensor(
                [self.registry.get(c, 0) for c in class_prompt],
                device=x.device
            )
            outputs['class_embeddings'] = self.class_embeddings(class_ids)
        
        return outputs
    
    def _split_labels(
        self,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split combined labels into semantic and instance masks using einops.
        
        Args:
            labels: Combined labels (B, 2, D, H, W)
            
        Returns:
            semantic: (B, 1, D, H, W)
            instance: (B, 1, D, H, W)
        """
        # Use einops to split channels cleanly
        semantic = rearrange(labels[:, 0], 'b d h w -> b 1 d h w')
        instance = rearrange(labels[:, 1], 'b d h w -> b 1 d h w')
        return semantic, instance
    
    def _create_neuron_mask(
        self,
        semantic_labels: torch.Tensor,
        neuron_id: int = 1
    ) -> torch.Tensor:
        """
        Create binary mask for neuron regions using einops.
        
        Args:
            semantic_labels: Semantic mask (B, 1, D, H, W)
            neuron_id: ID for neuron class
            
        Returns:
            Binary mask (B, 1, D, H, W)
        """
        return (semantic_labels == neuron_id).float()
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        
        # Split labels using einops helper
        labels_semantic, labels_instance = self._split_labels(batch['label'])
        
        # Forward
        outputs = self(images)
        semantic_logits = outputs['semantic_logits']
        embeddings = outputs['embeddings']
        
        # Semantic loss
        loss_semantic = self.sem_loss_fn(semantic_logits, labels_semantic.long())
        
        # Discriminative loss (restricted to neuron regions)
        neuron_id = self.registry.get('neuron', 1)
        neuron_mask = self._create_neuron_mask(labels_semantic, neuron_id)
        
        loss_disc, disc_dict = self.disc_loss_fn(
            embeddings,
            labels_instance,
            semantic_mask=neuron_mask
        )
        
        total_loss = loss_semantic + loss_disc
        
        # Logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_sem', loss_semantic, on_step=True, on_epoch=True)
        self.log('train/loss_disc', loss_disc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_var', disc_dict['loss_var'], on_step=False, on_epoch=True)
        self.log('train/loss_dist', disc_dict['loss_dist'], on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        images = batch['image']
        labels_semantic, labels_instance = self._split_labels(batch['label'])
        
        outputs = self(images)
        semantic_logits = outputs['semantic_logits']
        embeddings = outputs['embeddings']
        
        loss_semantic = self.sem_loss_fn(semantic_logits, labels_semantic.long())
        
        neuron_id = self.registry.get('neuron', 1)
        neuron_mask = self._create_neuron_mask(labels_semantic, neuron_id)
        
        loss_disc, _ = self.disc_loss_fn(
            embeddings,
            labels_instance,
            semantic_mask=neuron_mask
        )
        
        total_loss = loss_semantic + loss_disc
        
        self.log('val/loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val/loss_sem', loss_semantic, on_epoch=True)
        self.log('val/loss_disc', loss_disc, on_epoch=True)
        
        return total_loss
    
    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction with sliding window inference."""
        images = batch['image']
        patch_size = self.net_config.get('patch_size', (128, 128, 128))
        
        outputs = sliding_window_inference(
            inputs=images,
            roi_size=patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self._predict_fn,
            overlap=self.overlap,
            mode='gaussian'
        )
        
        return outputs
    
    def _predict_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Predictor for sliding window."""
        outputs = self(x)
        semantic_probs = torch.softmax(outputs['semantic_logits'], dim=1)
        
        # Concatenate using einops-style operation
        return torch.cat([semantic_probs, outputs['embeddings']], dim=1)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get('lr', 1e-4),
            weight_decay=self.optimizer_config.get('weight_decay', 1e-5)
        )
        
        scheduler_config = self.optimizer_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.optimizer_config.get('max_epochs', 100),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss'}
            }
        else:
            return optimizer
        
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            scheduler = self._create_warmup_scheduler(optimizer, scheduler, warmup_epochs)
        
        return [optimizer], [scheduler]
    
    def _create_warmup_scheduler(self, optimizer, main_scheduler, warmup_epochs: int):
        """Create scheduler with linear warmup."""
        from torch.optim.lr_scheduler import LambdaLR, SequentialLR
        
        warmup_scheduler = LambdaLR(
            optimizer,
            lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1.0
        )
        
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    
    def get_embedding_for_clustering(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings for post-hoc clustering.
        
        Returns:
            embeddings: (B, E, D, H, W)
            semantic_pred: (B, D, H, W)
        """
        with torch.no_grad():
            outputs = self(x)
            embeddings = outputs['embeddings']
            # Use reduce pattern for argmax
            semantic_pred = outputs['semantic_logits'].argmax(dim=1)
        
        return embeddings, semantic_pred
    
    def extract_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract intermediate features from backbone.
        
        Useful for transfer learning or feature analysis.
        """
        return self.backbone(x)


class VistaEmbedSmall(VistaLightningModule):
    """Lightweight VISTA-Embed for memory-constrained environments."""
    
    def _init_backbone(self):
        """Initialize smaller backbone."""
        self.backbone = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=self.net_config.get('in_channels', 1),
            out_channels=24,
            dropout_prob=self.net_config.get('dropout', 0.2),
            blocks_down=[1, 1, 2, 2],
            blocks_up=[1, 1]
        )
        self.net_config['feature_dim'] = 24


class VistaEmbedLarge(VistaLightningModule):
    """Larger VISTA-Embed for high-capacity training."""
    
    def _init_backbone(self):
        """Initialize larger backbone."""
        self.backbone = SegResNet(
            spatial_dims=3,
            init_filters=64,
            in_channels=self.net_config.get('in_channels', 1),
            out_channels=96,
            dropout_prob=self.net_config.get('dropout', 0.2),
            blocks_down=[2, 3, 4, 6],
            blocks_up=[2, 2, 2]
        )
        self.net_config['feature_dim'] = 96


class EmbeddingNormalizer(nn.Module):
    """
    Normalize embeddings using einops for clean operations.
    
    Options: L2 normalization, batch normalization, or none.
    """
    
    def __init__(self, method: str = 'l2', embedding_dim: int = 16):
        super().__init__()
        self.method = method
        
        if method == 'batch':
            self.norm = nn.BatchNorm3d(embedding_dim)
        elif method == 'instance':
            self.norm = nn.InstanceNorm3d(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize embeddings.
        
        Args:
            x: Embeddings (B, E, D, H, W)
        """
        if self.method == 'l2':
            # L2 normalize along embedding dimension using einops reduce
            norm = reduce(x ** 2, 'b e d h w -> b 1 d h w', 'sum').sqrt()
            norm = torch.clamp(norm, min=1e-8)
            return x / norm
        elif self.method in ['batch', 'instance']:
            return self.norm(x)
        else:
            return x
