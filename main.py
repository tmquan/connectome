"""
VISTA-Embed: Connectomics Instance Segmentation

Main entry point for training and inference using the VISTA3D-adapted
architecture for high-throughput connectomics segmentation.

This script follows the cosmed repository patterns:
- Hydra configuration management
- PyTorch Lightning training abstraction
- MONAI data pipelines

Usage:
    # Training with default config
    python main.py
    
    # Override specific parameters
    python main.py model.net_config.embedding_dim=32 training.max_epochs=200
    
    # Multi-GPU training
    python main.py training.devices=4 training.strategy=ddp
    
    # Run hyperparameter sweep
    python main.py --multirun model.net_config.embedding_dim=8,16,32
"""

import os
import sys
from pathlib import Path

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    ModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.models.vista_wrapper import VistaLightningModule
from src.data.datamodule import ConnectomicsDataModule, SNEMI3DDataModule
from src.utils.registry import LabelRegistry
from src.callbacks.visualization import VisualizationCallback, EmbeddingHistogramCallback


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup training callbacks from configuration."""
    callbacks = []
    
    callback_cfg = cfg.get('callbacks', {})
    
    # Model Checkpoint
    if 'checkpoint' in callback_cfg:
        ckpt_cfg = callback_cfg.checkpoint
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_cfg.get('dirpath', 'checkpoints'),
                filename=ckpt_cfg.get('filename', 'vista-{epoch:02d}-{train_loss_disc:.4f}'),
                save_top_k=ckpt_cfg.get('save_top_k', 3),
                monitor=ckpt_cfg.get('monitor', 'train/loss_disc'),
                mode=ckpt_cfg.get('mode', 'min'),
                save_last=ckpt_cfg.get('save_last', True),
                verbose=ckpt_cfg.get('verbose', True)
            )
        )
    
    # Early Stopping
    if 'early_stopping' in callback_cfg:
        es_cfg = callback_cfg.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get('monitor', 'val/loss'),
                patience=es_cfg.get('patience', 20),
                mode=es_cfg.get('mode', 'min'),
                verbose=es_cfg.get('verbose', True)
            )
        )
    
    # Learning Rate Monitor
    if 'lr_monitor' in callback_cfg:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=callback_cfg.lr_monitor.get('logging_interval', 'step')
            )
        )
    
    # Progress Bar
    callbacks.append(RichProgressBar())
    
    # Model Summary
    callbacks.append(ModelSummary(max_depth=3))
    
    # Visualization Callback (TensorBoard image logging)
    viz_cfg = callback_cfg.get('visualization', {})
    if viz_cfg.get('enabled', True):
        callbacks.append(
            VisualizationCallback(
                log_every_n_steps=viz_cfg.get('log_every_n_steps', 100),
                num_samples=viz_cfg.get('num_samples', 2),
                slice_idx=viz_cfg.get('slice_idx', None),
                embedding_viz_method=viz_cfg.get('embedding_viz_method', 'pca'),
                log_on_train=viz_cfg.get('log_on_train', True),
                log_on_val=viz_cfg.get('log_on_val', True)
            )
        )
    
    # Embedding Histogram Callback
    if viz_cfg.get('log_histograms', True):
        callbacks.append(
            EmbeddingHistogramCallback(
                log_every_n_steps=viz_cfg.get('histogram_every_n_steps', 500)
            )
        )
    
    return callbacks


def setup_logger(cfg: DictConfig):
    """Setup experiment logger."""
    logger_type = cfg.get('logger', 'tensorboard')
    
    if logger_type == 'tensorboard':
        return TensorBoardLogger(
            save_dir='logs',
            name=cfg.project_name,
            version=None
        )
    elif logger_type == 'wandb':
        return WandbLogger(
            project=cfg.project_name,
            name=f"{cfg.project_name}_{cfg.seed}",
            save_dir='logs'
        )
    else:
        return True  # Default Lightning logger


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main training entry point.
    
    Args:
        cfg: Hydra configuration object containing all parameters
    """
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    # 2. Initialize Label Registry
    registry = LabelRegistry.from_dict(dict(cfg.nlp_labels))
    print(f"\nLabel Registry: {registry.to_dict()}")
    print(f"Instance classes: {registry.get_instance_classes()}")
    print(f"Semantic classes: {registry.get_semantic_classes()}\n")
    
    # 3. Initialize DataModule
    # Use Hydra instantiation to support different datamodule types
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # 4. Initialize Model
    model = VistaLightningModule(
        net_config=dict(cfg.model.net_config),
        loss_config=dict(cfg.model.loss_config),
        optimizer_config=dict(cfg.model.optimizer_config),
        label_registry=registry.to_dict()
    )
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(f"  Backbone: SegResNet")
    print(f"  Init Filters: {cfg.model.net_config.init_filters}")
    print(f"  Feature Dim: {cfg.model.net_config.feature_dim}")
    print(f"  Embedding Dim: {cfg.model.net_config.embedding_dim}")
    print(f"  Semantic Classes: {len(registry.get_semantic_classes()) + 1}")
    
    # 5. Setup Callbacks
    callbacks = setup_callbacks(cfg)
    
    # 6. Setup Logger
    logger = setup_logger(cfg)
    
    # 7. Initialize Trainer
    training_cfg = cfg.training
    trainer = pl.Trainer(
        max_epochs=training_cfg.max_epochs,
        accelerator=training_cfg.accelerator,
        devices=training_cfg.devices,
        strategy=training_cfg.get('strategy', 'auto'),
        precision=training_cfg.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=training_cfg.log_every_n_steps,
        gradient_clip_val=training_cfg.gradient_clip_val,
        accumulate_grad_batches=training_cfg.get('accumulate_grad_batches', 1),
        val_check_interval=training_cfg.val_check_interval,
        check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,
        num_sanity_val_steps=training_cfg.num_sanity_val_steps,
        enable_progress_bar=training_cfg.enable_progress_bar,
        enable_model_summary=training_cfg.enable_model_summary,
        deterministic=training_cfg.get('deterministic', False),
        benchmark=training_cfg.get('benchmark', True),
        fast_dev_run=training_cfg.get('fast_dev_run', False)
    )
    
    # 8. Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    trainer.fit(model, datamodule)
    
    # 9. Save final checkpoint
    if trainer.global_rank == 0:
        final_path = os.path.join('checkpoints', 'final_model.ckpt')
        trainer.save_checkpoint(final_path)
        print(f"\nFinal model saved to: {final_path}")
    
    return trainer.callback_metrics


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def predict(cfg: DictConfig):
    """
    Inference entry point for processing new volumes.
    """
    from src.data.datamodule import InferenceDataModule
    from src.utils.clustering import cluster_embeddings
    
    # Load trained model
    checkpoint_path = cfg.get('checkpoint_path', 'checkpoints/final_model.ckpt')
    model = VistaLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Setup inference data
    volume_path = cfg.get('volume_path')
    if not volume_path:
        raise ValueError("Must specify volume_path for inference")
    
    inference_dm = InferenceDataModule(
        volume_path=volume_path,
        patch_size=tuple(cfg.data.patch_size) if hasattr(cfg.data.patch_size, '__iter__') 
                   else (cfg.data.patch_size,) * 3,
        overlap=cfg.data.get('patch_overlap', 0.5),
        batch_size=cfg.data.batch_size
    )
    
    # Run inference
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=1,
        precision=cfg.training.precision
    )
    
    predictions = trainer.predict(model, inference_dm)
    
    return predictions


if __name__ == "__main__":
    main()

