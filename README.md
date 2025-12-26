# VISTA-Embed: Connectomics Instance Segmentation

A PyTorch Lightning implementation adapting **VISTA3D** (Versatile Imaging SegmenTation and Annotation) foundation model for high-throughput electron microscopy (EM) connectomics segmentation.

## Overview

This repository provides an end-to-end pipeline for segmenting subcellular structures in EM connectomics data:
- **Neurons**: Individual neurites and somas (instance segmentation)
- **Mitochondria**: Organelles within neurons
- **Membranes**: Cell boundaries
- **Synapses**: Pre/post-synaptic structures

### Key Features

- **VISTA-Embed Architecture**: Modified VISTA3D backbone with dual decoder heads for semantic segmentation and instance embedding
- **Discriminative Loss**: Metric learning approach for separating densely packed neurons
- **Hydra Configuration**: Modular, reproducible experiment management
- **MONAI Data Pipeline**: Efficient handling of terabyte-scale volumetric data
- **Post-Processing**: Mean Shift, HDBSCAN, and Watershed clustering for instance generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VISTA-Embed Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Volume (B, 1, D, H, W)                              │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────────┐                                       │
│   │   SegResNet     │  ← Pre-trained encoder                │
│   │   Backbone      │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   Feature Map (B, F, D, H, W)                               │
│         │                                                    │
│    ┌────┴────┐                                              │
│    │         │                                               │
│    ▼         ▼                                               │
│ ┌──────┐ ┌──────────┐                                       │
│ │Semantic│ │Instance  │                                      │
│ │ Head   │ │  Head    │                                      │
│ └───┬────┘ └────┬─────┘                                     │
│     │           │                                            │
│     ▼           ▼                                            │
│  Logits    Embeddings                                        │
│(B,C,D,H,W) (B,E,D,H,W)                                      │
│     │           │                                            │
│     ▼           ▼                                            │
│  Dice/CE    Discriminative                                   │
│   Loss        Loss                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd connectome

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install pytest black isort flake8
```

## Project Structure

```
connectome/
├── conf/                          # Hydra configuration files
│   ├── config.yaml               # Master configuration
│   ├── model/
│   │   └── vista_connectomics.yaml
│   ├── data/
│   │   ├── snemi3d.yaml          # SNEMI3D neuron dataset
│   │   ├── mitoem2.yaml          # MitoEM2 mitochondria dataset
│   │   └── combined.yaml         # Combined multi-dataset training
│   ├── training/
│   │   └── default_trainer.yaml
│   └── callbacks/
│       └── checkpoint.yaml
├── src/
│   ├── models/
│   │   ├── vista_wrapper.py      # VistaLightningModule
│   │   ├── metrics.py            # ARI and clustering metrics
│   │   └── components/
│   │       └── discriminative.py # Discriminative Loss
│   ├── data/
│   │   ├── datamodule.py         # SNEMI3D, MitoEM2, CombinedDataModule
│   │   └── readers.py            # TIFF, H5, NIfTI readers
│   ├── callbacks/
│   │   └── visualization.py      # TensorBoard visualization
│   └── utils/
│       ├── registry.py           # Label Registry
│       └── clustering.py         # Post-processing
├── data/                         # Symlinks to datasets
│   ├── SNEMI3D/                  # -> /scratch/SNEMI3D/data
│   └── MitoEM2/                  # -> /scratch/MitoEM2
├── main.py                       # Training entry point
├── requirements.txt
└── README.md
```

## Data Preparation

### Supported Formats

The pipeline supports multiple data formats with automatic detection:

| Format | Extensions | Description |
|--------|------------|-------------|
| TIFF | `.tiff`, `.tif` | Multi-page 3D volumes (SNEMI3D) |
| NIfTI | `.nii.gz`, `.nii` | Medical imaging format (MitoEM2) |
| HDF5 | `.h5`, `.hdf5` | Hierarchical data format |

### SNEMI3D Format

```
data/SNEMI3D/
├── AC3_inputs.tiff    # EM images (100, 1024, 1024) uint8
├── AC3_labels.tiff    # Instance labels (100, 1024, 1024) uint16
├── AC4_inputs.tiff
└── AC4_labels.tiff
```

### MitoEM2 Format (nnUNet-style)

```
data/MitoEM2/
└── Dataset001_ME2-Beta/
    ├── imagesTr/
    │   └── me2-beta_train01_0000.nii.gz
    └── labelsTr/
        └── me2-beta_train01.nii.gz
```

### Label Format

Labels are instance segmentation masks:
- **Value 0**: Background
- **Value 1+**: Unique instance IDs (e.g., individual neurons or mitochondria)

The semantic class is assigned based on the dataset configuration (`semantic_class_id`).

## Usage

### Training

```bash
# Basic training (SNEMI3D neurons only)
python main.py

# Train with MitoEM2 mitochondria only
python main.py data=mitoem2

# Train with combined datasets (neurons + mitochondria)
python main.py data=combined

# Override parameters
python main.py \
    model.net_config.embedding_dim=32 \
    training.max_epochs=200 \
    data.batch_size=4

# Multi-GPU training with combined datasets
python main.py \
    data=snemi3d \
    data.batch_size=1 \
    training.devices=4 \
    training.strategy=ddp_find_unused_parameters_true

# Hyperparameter sweep
python main.py --multirun \
    model.net_config.embedding_dim=8,16,32 \
    model.loss_config.discriminative.delta_var=0.3,0.5,0.7
```

### Supported Datasets

| Dataset | Data Config | Semantic Class | Description |
|---------|-------------|----------------|-------------|
| SNEMI3D | `data=snemi3d` | neuron (1) | Neurite instance segmentation |
| MitoEM2 | `data=mitoem2` | mitochondria (2) | Mitochondria in 8 cell types |
| Combined | `data=combined` | both | Multi-task training |

### Inference

```python
from src.models import VistaLightningModule
from src.utils import cluster_embeddings

# Load trained model
model = VistaLightningModule.load_from_checkpoint("checkpoints/final_model.ckpt")
model.eval()

# Get embeddings
with torch.no_grad():
    outputs = model(volume)
    embeddings = outputs['embeddings']
    semantic_pred = torch.argmax(outputs['semantic_logits'], dim=1)

# Cluster to instances
neuron_mask = semantic_pred == 1  # Neuron class
instance_mask = cluster_embeddings(
    embeddings.cpu().numpy(),
    foreground_mask=neuron_mask.cpu().numpy(),
    method='mean_shift',
    bandwidth=0.5
)
```

## Configuration Reference

### Model Configuration (`conf/model/vista_connectomics.yaml`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `net_config.init_filters` | Initial convolution filters | 32 |
| `net_config.feature_dim` | Backbone output dimension | 48 |
| `net_config.embedding_dim` | Instance embedding dimension | 16 |
| `loss_config.discriminative.delta_var` | Variance loss margin | 0.5 |
| `loss_config.discriminative.delta_dist` | Distance loss margin | 1.5 |

### Data Configuration (`conf/data/snemi3d.yaml`, `combined.yaml`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `patch_size` | Training patch dimensions (D, H, W) | [32, 256, 256] |
| `batch_size` | Training batch size | 2 |
| `patches_per_volume` | Patches sampled per epoch per volume | 100 |
| `cache_volumes` | Cache volumes in memory | true |
| `semantic_class_id` | Semantic class ID for dataset | 1 (neuron) |

### Combined Dataset Configuration (`conf/data/combined.yaml`)

```yaml
datasets:
  - type: snemi3d
    data_root: "data/SNEMI3D/"
    train_volumes: ["AC3"]
    val_volumes: ["AC4"]
    semantic_class_id: 1  # neuron
  
  - type: mitoem2
    data_root: "data/MitoEM2/"
    datasets:
      - "Dataset001_ME2-Beta"
      - "Dataset006_ME2-Pyra"
    semantic_class_id: 2  # mitochondria
```

## Discriminative Loss

The model uses the Discriminative Loss function for instance embedding:

$$\mathcal{L} = \alpha \mathcal{L}_{var} + \beta \mathcal{L}_{dist} + \gamma \mathcal{L}_{reg}$$

Where:
- **Variance Loss** ($\mathcal{L}_{var}$): Pulls embeddings towards their instance mean
- **Distance Loss** ($\mathcal{L}_{dist}$): Pushes instance means apart
- **Regularization Loss** ($\mathcal{L}_{reg}$): Keeps means near origin

## Label Registry

The label registry maps biological entity names to VISTA3D's embedding lookup indices:

```python
from src.utils import LabelRegistry

# Create default registry
registry = LabelRegistry.default()

# Map name to ID (VISTA3D's integer-to-embedding mechanism)
neuron_id = registry.name_to_id('neuron')  # Returns 1

# Get instance vs semantic classes
instance_classes = registry.get_instance_classes()  # ['neuron']
semantic_classes = registry.get_semantic_classes()  # ['mitochondria', 'membrane', 'synapse']
```

## Post-Processing Clustering

After training, convert embeddings to instance masks:

```python
from src.utils.clustering import (
    mean_shift_clustering,
    hdbscan_clustering,
    watershed_from_embeddings
)

# Mean Shift (recommended)
instances = mean_shift_clustering(
    embeddings,
    foreground_mask,
    bandwidth=0.5,
    min_bin_freq=100
)

# HDBSCAN (for varying density)
instances = hdbscan_clustering(
    embeddings,
    foreground_mask,
    min_cluster_size=100
)

# Watershed (fastest)
instances = watershed_from_embeddings(
    embeddings,
    foreground_mask,
    seed_threshold=0.3
)
```

## References

1. **VISTA3D**: He, Y., et al. "VISTA3D: A Unified Segmentation Foundation Model for 3D Medical Imaging." (2024)
2. **Discriminative Loss**: De Brabandere, B., et al. "Semantic Instance Segmentation with a Discriminative Loss Function." arXiv:1708.02551 (2017)
3. **SegResNet**: Myronenko, A. "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization." BrainLes@MICCAI (2018)
4. **MONAI**: MONAI Consortium. "MONAI: Medical Open Network for AI." (2020)

## License

MIT License

## Citation

```bibtex
@software{vista_embed_connectomics,
  title = {VISTA-Embed: Adapting VISTA3D for Connectomics Instance Segmentation},
  year = {2024},
  publisher = {GitHub},
  url = {<repository-url>}
}
```
