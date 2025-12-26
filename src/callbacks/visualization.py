"""
TensorBoard Visualization Callback

Logs input images, output embeddings, predictions, and targets
to TensorBoard for monitoring training progress.

Uses einops for tensor manipulation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any, List
from einops import rearrange, reduce, repeat
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import label as scipy_label


def cluster_embeddings_simple(
    embeddings: torch.Tensor,
    semantic_mask: torch.Tensor,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50
) -> torch.Tensor:
    """
    Simple mean-shift-like clustering for instance segmentation visualization.
    
    Fast approximation suitable for visualization during training.
    
    Args:
        embeddings: (E, H, W) embedding tensor
        semantic_mask: (H, W) binary foreground mask
        bandwidth: Distance threshold for clustering
        min_cluster_size: Minimum pixels per instance
        
    Returns:
        (H, W) instance segmentation tensor
    """
    E, H, W = embeddings.shape
    device = embeddings.device
    
    # Get foreground pixels
    fg_mask = semantic_mask > 0
    if fg_mask.sum() == 0:
        return torch.zeros(H, W, device=device, dtype=torch.long)
    
    # Flatten embeddings for foreground pixels: (N, E)
    fg_indices = torch.where(fg_mask)
    fg_embeddings = embeddings[:, fg_indices[0], fg_indices[1]].T  # (N, E)
    
    # Normalize embeddings
    fg_embeddings = F.normalize(fg_embeddings, dim=1)
    
    # Simple greedy clustering
    N = fg_embeddings.shape[0]
    labels = torch.zeros(N, device=device, dtype=torch.long)
    current_label = 1
    
    # Subsample if too many points (for speed)
    max_seeds = 100
    if N > max_seeds * 10:
        seed_indices = torch.randperm(N, device=device)[:max_seeds]
    else:
        seed_indices = torch.arange(0, N, max(1, N // max_seeds), device=device)
    
    for seed_idx in seed_indices:
        if labels[seed_idx] > 0:
            continue
        
        seed_emb = fg_embeddings[seed_idx:seed_idx+1]  # (1, E)
        
        # Find all points within bandwidth
        distances = 1 - torch.mm(fg_embeddings, seed_emb.T).squeeze()  # Cosine distance
        cluster_mask = (distances < bandwidth) & (labels == 0)
        
        if cluster_mask.sum() >= min_cluster_size:
            labels[cluster_mask] = current_label
            current_label += 1
    
    # Create output image
    instance_map = torch.zeros(H, W, device=device, dtype=torch.long)
    instance_map[fg_indices[0], fg_indices[1]] = labels
    
    return instance_map


class VisualizationCallback(Callback):
    """
    TensorBoard callback for visualizing connectomics segmentation.
    
    Logs:
    - Input EM images
    - Output embeddings (as RGB using PCA or first 3 channels)
    - Output semantic predictions
    - Target labels (semantic + instance)
    
    Args:
        log_every_n_steps: Log visualizations every N training steps
        num_samples: Number of samples to visualize per batch
        slice_idx: Which z-slice to visualize (None = middle)
        embedding_viz_method: 'pca' or 'channels' for embedding visualization
        log_on_train: Whether to log during training
        log_on_val: Whether to log during validation
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 100,
        num_samples: int = 2,
        slice_idx: Optional[int] = None,
        embedding_viz_method: str = 'pca',
        log_on_train: bool = True,
        log_on_val: bool = True,
        max_instances_colormap: int = 50
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples
        self.slice_idx = slice_idx
        self.embedding_viz_method = embedding_viz_method
        self.log_on_train = log_on_train
        self.log_on_val = log_on_val
        self.max_instances_colormap = max_instances_colormap
        
        # Color map for instance visualization
        self._instance_colors = self._generate_instance_colormap()
    
    def _generate_instance_colormap(self) -> torch.Tensor:
        """Generate distinct colors for instance visualization."""
        np.random.seed(42)  # Reproducible colors
        colors = np.random.rand(self.max_instances_colormap, 3)
        colors[0] = [0, 0, 0]  # Background is black
        return torch.from_numpy(colors).float()
    
    def _get_slice(self, volume: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """
        Extract a 2D slice from a volume.
        
        Args:
            volume: (B, C, D, H, W) or (B, D, H, W)
            dim: Spatial dimension to slice (2=D, 3=H, 4=W for 5D)
        """
        if volume.ndim == 5:
            # (B, C, D, H, W)
            d = volume.shape[2]
            idx = self.slice_idx if self.slice_idx is not None else d // 2
            idx = min(idx, d - 1)
            return volume[:, :, idx, :, :]  # (B, C, H, W)
        elif volume.ndim == 4:
            # (B, D, H, W)
            d = volume.shape[1]
            idx = self.slice_idx if self.slice_idx is not None else d // 2
            idx = min(idx, d - 1)
            return volume[:, idx, :, :]  # (B, H, W)
        else:
            return volume
    
    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] range."""
        img = img.float()
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img
    
    def _embedding_to_rgb(
        self, 
        embedding: torch.Tensor,
        method: str = 'pca'
    ) -> torch.Tensor:
        """
        Convert high-dimensional embedding to RGB image.
        
        Args:
            embedding: (B, E, H, W) embedding tensor
            method: 'pca' or 'channels'
            
        Returns:
            (B, 3, H, W) RGB tensor
        """
        B, E, H, W = embedding.shape
        
        if method == 'channels':
            # Use first 3 channels directly
            if E >= 3:
                rgb = embedding[:, :3, :, :]
            else:
                # Pad with zeros if less than 3 channels
                rgb = F.pad(embedding, (0, 0, 0, 0, 0, 3 - E))
        else:
            # PCA reduction to 3 components
            try:
                # Flatten: (B, E, H, W) -> (B*H*W, E)
                flat = rearrange(embedding, 'b e h w -> (b h w) e')
                
                # Center the data
                mean = flat.mean(dim=0, keepdim=True)
                centered = flat - mean
                
                # Compute covariance: (E, E)
                cov = torch.mm(centered.T, centered) / centered.shape[0]
                
                # SVD: Vh has shape (E, E), rows are eigenvectors
                _, _, Vh = torch.linalg.svd(cov)
                
                # Take first 3 eigenvectors: (3, E)
                components = Vh[:3, :]
                
                # Project: (B*H*W, E) @ (3, E).T -> (B*H*W, 3)
                projected = torch.mm(centered, components.T)
                
                # Reshape: (B*H*W, 3) -> (B, 3, H, W)
                rgb = rearrange(projected, '(b h w) c -> b c h w', b=B, h=H, w=W)
                
            except Exception:
                # Fallback to first 3 channels
                if E >= 3:
                    rgb = embedding[:, :3, :, :].clone()
                else:
                    rgb = F.pad(embedding.clone(), (0, 0, 0, 0, 0, 3 - E))
        
        # Normalize each channel to [0, 1]
        rgb = rgb.clone()
        for c in range(3):
            channel = rgb[:, c, :, :]
            c_min, c_max = channel.min(), channel.max()
            if c_max > c_min:
                rgb[:, c, :, :] = (channel - c_min) / (c_max - c_min)
            else:
                rgb[:, c, :, :] = 0.5
        
        return rgb
    
    def _semantic_to_rgb(
        self, 
        semantic: torch.Tensor,
        num_classes: int = 5
    ) -> torch.Tensor:
        """
        Convert semantic prediction to colored image.
        
        Args:
            semantic: (B, H, W) class indices
            
        Returns:
            (B, 3, H, W) RGB tensor
        """
        # Define semantic colormap
        # 0: background (black), 1: neuron (red), 2: mito (green), 
        # 3: membrane (blue), 4: synapse (yellow)
        colors = torch.tensor([
            [0.0, 0.0, 0.0],    # background - black
            [1.0, 1.0, 1.0],    # neuron - white
            [0.2, 1.0, 0.2],    # mitochondria - green
            [0.2, 0.2, 1.0],    # membrane - blue
            [1.0, 1.0, 0.2],    # synapse - yellow
        ], device=semantic.device)
        
        # Clamp to valid range
        semantic = semantic.long().clamp(0, len(colors) - 1)
        
        # Map to colors: (B, H, W) -> (B, H, W, 3) -> (B, 3, H, W)
        rgb = colors[semantic]
        rgb = rearrange(rgb, 'b h w c -> b c h w')
        
        return rgb
    
    def _instance_to_rgb(
        self, 
        instance: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert instance mask to colored image.
        
        Args:
            instance: (B, H, W) instance IDs
            
        Returns:
            (B, 3, H, W) RGB tensor
        """
        colors = self._instance_colors.to(instance.device)
        
        # Map instance IDs to color indices (mod to handle many instances)
        color_idx = (instance.long() % self.max_instances_colormap)
        
        # Map to colors
        rgb = colors[color_idx]
        rgb = rearrange(rgb, 'b h w c -> b c h w')
        
        return rgb
    
    def _cluster_embeddings_to_instances(
        self,
        embeddings: torch.Tensor,
        semantic_pred: torch.Tensor,
        bandwidth: float = 0.5
    ) -> torch.Tensor:
        """
        Cluster embeddings to produce instance segmentation.
        
        Args:
            embeddings: (B, E, H, W) embedding tensor
            semantic_pred: (B, H, W) semantic prediction (foreground mask)
            bandwidth: Clustering bandwidth
            
        Returns:
            (B, H, W) instance segmentation
        """
        B = embeddings.shape[0]
        instances = []
        
        for b in range(B):
            emb = embeddings[b]  # (E, H, W)
            sem = semantic_pred[b]  # (H, W)
            inst = cluster_embeddings_simple(emb, sem, bandwidth=bandwidth)
            instances.append(inst)
        
        return torch.stack(instances, dim=0)
    
    def _log_images(
        self,
        logger: SummaryWriter,
        tag_prefix: str,
        images: torch.Tensor,
        embeddings: torch.Tensor,
        semantic_pred: torch.Tensor,
        instance_pred: torch.Tensor,
        semantic_target: torch.Tensor,
        instance_target: torch.Tensor,
        global_step: int
    ):
        """Log all visualizations to TensorBoard."""
        # Limit samples
        n = min(self.num_samples, images.shape[0])
        
        # Extract slices from 3D volumes
        img_slice = self._get_slice(images[:n])  # (n, 1, H, W)
        emb_slice = self._get_slice(embeddings[:n])  # (n, E, H, W)
        sem_pred_slice = self._get_slice(semantic_pred[:n])  # (n, H, W)
        inst_pred_slice = self._get_slice(instance_pred[:n])  # (n, H, W)
        sem_tgt_slice = self._get_slice(semantic_target[:n])  # (n, H, W)
        inst_tgt_slice = self._get_slice(instance_target[:n])  # (n, H, W)
        
        # Convert to RGB visualizations - all should be (n, 3, H, W)
        img_rgb = repeat(self._normalize_image(img_slice), 'b 1 h w -> b 3 h w')
        emb_rgb = self._embedding_to_rgb(emb_slice, self.embedding_viz_method)
        sem_pred_rgb = self._semantic_to_rgb(sem_pred_slice)
        inst_pred_rgb = self._instance_to_rgb(inst_pred_slice)
        sem_tgt_rgb = self._semantic_to_rgb(sem_tgt_slice.long())
        inst_tgt_rgb = self._instance_to_rgb(inst_tgt_slice)
        
        # Ensure all tensors have same spatial dimensions
        H, W = img_rgb.shape[2], img_rgb.shape[3]
        
        # Resize if needed (in case embedding output has different resolution)
        if emb_rgb.shape[2:] != (H, W):
            emb_rgb = F.interpolate(emb_rgb, size=(H, W), mode='bilinear', align_corners=False)
        if sem_pred_rgb.shape[2:] != (H, W):
            sem_pred_rgb = F.interpolate(sem_pred_rgb, size=(H, W), mode='nearest')
        if inst_pred_rgb.shape[2:] != (H, W):
            inst_pred_rgb = F.interpolate(inst_pred_rgb, size=(H, W), mode='nearest')
        if sem_tgt_rgb.shape[2:] != (H, W):
            sem_tgt_rgb = F.interpolate(sem_tgt_rgb, size=(H, W), mode='nearest')
        if inst_tgt_rgb.shape[2:] != (H, W):
            inst_tgt_rgb = F.interpolate(inst_tgt_rgb, size=(H, W), mode='nearest')
        
        # Create single grid with all samples
        # Columns: Input | Embedding | SemPred | InstPred | SemTarget | InstTarget
        rows = []
        for i in range(n):
            row = torch.cat([
                img_rgb[i],
                emb_rgb[i],
                sem_pred_rgb[i],
                inst_pred_rgb[i],
                sem_tgt_rgb[i],
                inst_tgt_rgb[i]
            ], dim=2)  # Concatenate along width
            rows.append(row)
        
        # Stack all rows vertically
        grid = torch.cat(rows, dim=1)
        
        logger.add_image(
            f'{tag_prefix}/visualization',
            grid,
            global_step
        )
        
        # # Also log individual components as grids
        # from torchvision.utils import make_grid
        
        # logger.add_image(
        #     f'{tag_prefix}/inputs',
        #     make_grid(img_rgb, nrow=n, normalize=False),
        #     global_step
        # )
        
        # logger.add_image(
        #     f'{tag_prefix}/embeddings',
        #     make_grid(emb_rgb, nrow=n, normalize=False),
        #     global_step
        # )
        
        # logger.add_image(
        #     f'{tag_prefix}/predictions',
        #     make_grid(sem_pred_rgb, nrow=n, normalize=False),
        #     global_step
        # )
        
        # logger.add_image(
        #     f'{tag_prefix}/targets_semantic',
        #     make_grid(sem_tgt_rgb, nrow=n, normalize=False),
        #     global_step
        # )
        
        # logger.add_image(
        #     f'{tag_prefix}/targets_instance',
        #     make_grid(inst_tgt_rgb, nrow=n, normalize=False),
        #     global_step
        # )
    
    def _is_main_process(self, trainer: pl.Trainer) -> bool:
        """Check if this is the main process (rank 0) for logging."""
        # Only log on rank 0 to avoid NCCL synchronization issues
        if hasattr(trainer, 'global_rank'):
            return trainer.global_rank == 0
        return True
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Log visualizations during training."""
        if not self.log_on_train:
            return
        
        # Only run on main process to avoid DDP sync issues
        if not self._is_main_process(trainer):
            return
        
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        if trainer.logger is None:
            return
        
        # Get tensorboard logger
        tb_logger = self._get_tensorboard_logger(trainer)
        if tb_logger is None:
            return
        
        self._log_batch(
            pl_module, batch, tb_logger,
            'train', trainer.global_step
        )
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Log visualizations during validation."""
        if not self.log_on_val:
            return
        
        # Only run on main process to avoid DDP sync issues
        if not self._is_main_process(trainer):
            return
        
        # Only log first batch
        if batch_idx != 0:
            return
        
        if trainer.logger is None:
            return
        
        tb_logger = self._get_tensorboard_logger(trainer)
        if tb_logger is None:
            return
        
        self._log_batch(
            pl_module, batch, tb_logger,
            'val', trainer.global_step
        )
    
    def _get_tensorboard_logger(self, trainer: pl.Trainer) -> Optional[SummaryWriter]:
        """Extract TensorBoard SummaryWriter from trainer."""
        if hasattr(trainer.logger, 'experiment'):
            exp = trainer.logger.experiment
            if isinstance(exp, SummaryWriter):
                return exp
        
        # Handle multiple loggers
        if hasattr(trainer, 'loggers'):
            for logger in trainer.loggers:
                if hasattr(logger, 'experiment'):
                    exp = logger.experiment
                    if isinstance(exp, SummaryWriter):
                        return exp
        
        return None
    
    @torch.no_grad()
    def _log_batch(
        self,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        logger: SummaryWriter,
        tag_prefix: str,
        global_step: int
    ):
        """Process a batch and log visualizations."""
        images = batch['image']
        labels = batch['label']
        
        # Extract semantic and instance targets
        # Labels format: (B, 2, D, H, W) where [0]=semantic, [1]=instance
        semantic_target = labels[:, 0, ...]  # (B, D, H, W)
        instance_target = labels[:, 1, ...]  # (B, D, H, W)
        
        # Forward pass
        pl_module.eval()
        outputs = pl_module(images)
        pl_module.train()
        
        # Get predictions
        embeddings = outputs['embeddings']  # (B, E, D, H, W)
        semantic_logits = outputs['semantic_logits']  # (B, C, D, H, W)
        semantic_pred = semantic_logits.argmax(dim=1)  # (B, D, H, W)
        
        # Cluster embeddings to get instance predictions
        # Use middle slice for clustering (faster than full 3D)
        B, E, D, H, W = embeddings.shape
        slice_idx = self.slice_idx if self.slice_idx is not None else D // 2
        slice_idx = min(slice_idx, D - 1)
        
        emb_slice = embeddings[:, :, slice_idx, :, :]  # (B, E, H, W)
        sem_slice = semantic_pred[:, slice_idx, :, :]  # (B, H, W)
        
        # Cluster to get instance predictions
        instance_pred_slice = self._cluster_embeddings_to_instances(
            emb_slice, sem_slice, bandwidth=0.5
        )  # (B, H, W)
        
        # Expand back to 3D for consistent interface (only middle slice has data)
        instance_pred = torch.zeros_like(semantic_pred)
        instance_pred[:, slice_idx, :, :] = instance_pred_slice
        
        # Log
        self._log_images(
            logger=logger,
            tag_prefix=tag_prefix,
            images=images,
            embeddings=embeddings,
            semantic_pred=semantic_pred,
            instance_pred=instance_pred,
            semantic_target=semantic_target,
            instance_target=instance_target,
            global_step=global_step
        )


class EmbeddingHistogramCallback(Callback):
    """
    Log embedding statistics to TensorBoard.
    
    Tracks:
    - Embedding distribution histograms
    - Embedding norms
    - Inter-instance distances
    """
    
    def __init__(self, log_every_n_steps: int = 500):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        # Only run on main process to avoid DDP sync issues
        if hasattr(trainer, 'global_rank') and trainer.global_rank != 0:
            return
        
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        if trainer.logger is None:
            return
        
        # Get tensorboard logger
        if not hasattr(trainer.logger, 'experiment'):
            return
        
        logger = trainer.logger.experiment
        
        with torch.no_grad():
            images = batch['image']
            outputs = pl_module(images)
            embeddings = outputs['embeddings']
            
            # Flatten embeddings
            flat_emb = rearrange(embeddings, 'b e d h w -> (b d h w) e')
            
            # Log embedding dimension histograms
            for i in range(min(embeddings.shape[1], 4)):  # First 4 dims
                logger.add_histogram(
                    f'embeddings/dim_{i}',
                    flat_emb[:, i],
                    trainer.global_step
                )
            
            # Log embedding norms
            norms = torch.norm(flat_emb, dim=1)
            logger.add_histogram('embeddings/norms', norms, trainer.global_step)
            logger.add_scalar('embeddings/mean_norm', norms.mean(), trainer.global_step)

