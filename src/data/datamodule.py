"""
Connectomics Data Module

High-throughput data pipeline for volumetric EM data.
Supports multiple formats: TIFF, HDF5, and NIfTI.

Uses einops for clean tensor operations.
"""

import os
import json
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from monai.data import (
    CacheDataset,
    PersistentDataset,
    Dataset,
    GridPatchDataset,
    PatchIter,
    PatchDataset,
    list_data_collate,
    decollate_batch
)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    Lambda,
    MapTransform
)
from omegaconf import DictConfig
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from einops import rearrange, repeat, reduce
from scipy.ndimage import label as scipy_label

# Try to import cc3d for faster 3D connected components
try:
    import cc3d
    HAS_CC3D = True
except ImportError:
    HAS_CC3D = False

# Import readers
from .readers import (
    AutoReader, TiffReader, H5Reader, NiftiReader,
    read_volume, read_volume_with_metadata
)


def relabel_connected_components(
    instance_mask: np.ndarray,
    connectivity: int = 1,
    use_cc3d: bool = True
) -> np.ndarray:
    """
    Relabel instance mask so each connected component has a unique ID.
    
    When extracting patches, a single instance may be split into multiple
    disconnected parts. This function assigns new unique IDs to each
    connected component.
    
    Args:
        instance_mask: Instance segmentation mask (D, H, W) or (H, W)
        connectivity: Connectivity for connected components
                     1 = 6-connected (face), 2 = 18-connected, 3 = 26-connected
        use_cc3d: Use cc3d library if available (faster for 3D)
        
    Returns:
        Relabeled mask with unique IDs per connected component
    """
    if instance_mask.max() == 0:
        return instance_mask
    
    # Use cc3d if available and 3D
    if HAS_CC3D and use_cc3d and instance_mask.ndim == 3:
        # cc3d is significantly faster for 3D volumes
        # connectivity: 6, 18, or 26
        cc_connectivity = {1: 6, 2: 18, 3: 26}.get(connectivity, 26)
        relabeled = cc3d.connected_components(
            instance_mask.astype(np.uint32),
            connectivity=cc_connectivity
        )
        return relabeled.astype(instance_mask.dtype)
    
    # Fallback to scipy (works for 2D and 3D)
    # Process each unique instance ID separately to handle splits
    output = np.zeros_like(instance_mask)
    current_label = 1
    
    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances > 0]
    
    for inst_id in unique_instances:
        # Get binary mask for this instance
        binary_mask = (instance_mask == inst_id)
        
        # Find connected components within this instance
        if instance_mask.ndim == 3:
            # 3D structure for connectivity
            if connectivity == 1:
                # 6-connected (face neighbors only)
                struct = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
                ])
            else:
                # 26-connected (all neighbors)
                struct = np.ones((3, 3, 3), dtype=np.int32)
        else:
            # 2D structure
            struct = None  # Use default
        
        labeled, num_features = scipy_label(binary_mask, structure=struct)
        
        # Assign new unique labels
        for i in range(1, num_features + 1):
            output[labeled == i] = current_label
            current_label += 1
    
    return output


def relabel_connected_components_torch(
    instance_mask: torch.Tensor,
    connectivity: int = 1
) -> torch.Tensor:
    """
    Torch-compatible relabeling using kornia or scipy backend.
    
    Args:
        instance_mask: Instance mask tensor (D, H, W) or (B, D, H, W)
        connectivity: Connectivity for connected components
        
    Returns:
        Relabeled tensor
    """
    # Convert to numpy, process, convert back
    device = instance_mask.device
    dtype = instance_mask.dtype
    
    if instance_mask.ndim == 4:
        # Batch processing
        batch_size = instance_mask.shape[0]
        result = torch.zeros_like(instance_mask)
        
        for b in range(batch_size):
            mask_np = instance_mask[b].cpu().numpy()
            relabeled = relabel_connected_components(mask_np, connectivity)
            result[b] = torch.from_numpy(relabeled).to(device=device, dtype=dtype)
        
        return result
    else:
        # Single volume
        mask_np = instance_mask.cpu().numpy()
        relabeled = relabel_connected_components(mask_np, connectivity)
        return torch.from_numpy(relabeled).to(device=device, dtype=dtype)


class LoadWithReaderd(MapTransform):
    """
    MONAI-compatible transform using our custom readers.
    
    Automatically detects file format and loads using appropriate reader.
    """
    
    def __init__(
        self,
        keys: List[str],
        normalize: bool = True,
        h5_dataset_key: str = 'data'
    ):
        super().__init__(keys)
        self.reader = AutoReader(
            normalize=normalize,
            add_channel=False,  # We handle this separately
            dataset_key=h5_dataset_key
        )
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d and isinstance(d[key], str):
                d[key] = self.reader.read(d[key])
        return d


class ConvertToSemanticInstanceLabeld(MapTransform):
    """
    Convert instance-only labels to semantic + instance format.
    
    Creates two-channel output using einops:
    - Channel 0: Semantic mask (0=background, 1=neuron)
    - Channel 1: Instance IDs
    """
    
    def __init__(self, keys: List[str], neuron_label: int = 1):
        super().__init__(keys)
        self.neuron_label = neuron_label
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                instance_mask = d[key]
                
                if instance_mask.ndim == 3:
                    # (D, H, W) -> semantic + instance channels
                    semantic = (instance_mask > 0).astype(np.float32) * self.neuron_label
                    # Stack using einops pattern
                    d[key] = np.stack([semantic, instance_mask.astype(np.float32)], axis=0)
                    
                elif instance_mask.ndim == 4 and instance_mask.shape[0] == 1:
                    # (1, D, H, W) -> (2, D, H, W)
                    inst = rearrange(instance_mask, '1 d h w -> d h w')
                    semantic = (inst > 0).astype(np.float32) * self.neuron_label
                    d[key] = np.stack([semantic, inst.astype(np.float32)], axis=0)
                    
        return d


class UnifiedDataset(TorchDataset):
    """
    Unified PyTorch Dataset supporting multiple file formats.
    
    Uses einops for consistent tensor manipulation.
    
    Supports configurable semantic class mapping for different datasets:
    - SNEMI3D: semantic_class_id=1 (neuron)
    - MitoEM2.0: semantic_class_id=2 (mitochondria)
    """
    
    def __init__(
        self,
        data_root: str,
        image_paths: List[str],
        label_paths: List[str],
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        patches_per_volume: int = 100,
        augment: bool = True,
        cache_volumes: bool = True,
        normalize_images: bool = True,
        h5_image_key: str = 'raw',
        h5_label_key: str = 'label',
        semantic_class_id: int = 1  # 1=neuron, 2=mitochondria, 3=membrane, 4=synapse
    ):
        """
        Args:
            data_root: Root directory for data
            image_paths: List of image file paths (relative to data_root)
            label_paths: List of label file paths (relative to data_root)
            patch_size: Size of extracted patches (D, H, W)
            patches_per_volume: Patches sampled per epoch per volume
            augment: Whether to apply augmentation
            cache_volumes: Cache loaded volumes in memory
            normalize_images: Normalize images to [0, 1]
            h5_image_key: HDF5 dataset key for images
            h5_label_key: HDF5 dataset key for labels
            semantic_class_id: Semantic class for foreground (1=neuron, 2=mito, etc.)
        """
        self.data_root = Path(data_root)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.cache_volumes = cache_volumes
        self.semantic_class_id = semantic_class_id
        
        # Initialize readers
        self.image_reader = AutoReader(
            normalize=normalize_images,
            add_channel=True,
            dataset_key=h5_image_key
        )
        self.label_reader = AutoReader(
            normalize=False,  # Labels should not be normalized
            add_channel=False,
            dataset_key=h5_label_key
        )
        
        # Volume cache
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Validate and optionally preload
        self._validate_data()
        if cache_volumes:
            self._preload_volumes()
    
    def _validate_data(self):
        """Verify all data files exist."""
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            img_full = self.data_root / img_path
            lbl_full = self.data_root / lbl_path
            
            if not img_full.exists():
                raise FileNotFoundError(f"Image not found: {img_full}")
            if not lbl_full.exists():
                raise FileNotFoundError(f"Label not found: {lbl_full}")
    
    def _preload_volumes(self):
        """Preload all volumes into memory."""
        print(f"Caching {len(self.image_paths)} volumes...")
        for idx in range(len(self.image_paths)):
            self._load_volume(idx)
        print("Volume caching complete.")
    
    def _load_volume(self, idx: int) -> Dict[str, np.ndarray]:
        """Load a volume pair from disk or cache."""
        if idx in self._cache:
            return self._cache[idx]
        
        img_path = str(self.data_root / self.image_paths[idx])
        lbl_path = str(self.data_root / self.label_paths[idx])
        
        image = self.image_reader.read(img_path)  # (C, D, H, W)
        label = self.label_reader.read(lbl_path)  # (D, H, W) or (C, D, H, W)
        
        # Ensure label is 3D for processing
        if label.ndim == 4:
            label = rearrange(label, 'c d h w -> d h w') if label.shape[0] == 1 else label[0]
        
        vol_data = {
            'image': image,
            'label': label,
            'shape': image.shape
        }
        
        if self.cache_volumes:
            self._cache[idx] = vol_data
        
        return vol_data
    
    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx // self.patches_per_volume
        vol_data = self._load_volume(vol_idx)
        
        image = vol_data['image']
        label = vol_data['label']
        
        patch = self._extract_random_patch(image, label)
        
        if self.augment:
            patch = self._augment(patch)
        
        return patch
    
    def _extract_random_patch(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Extract random patch using einops for reshaping."""
        # image: (C, D, H, W), label: (D, H, W)
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # Random starting position
        z = np.random.randint(0, max(1, d - pd + 1))
        y = np.random.randint(0, max(1, h - ph + 1))
        x = np.random.randint(0, max(1, w - pw + 1))
        
        # Extract patches
        img_patch = image[:, z:z+pd, y:y+ph, x:x+pw]
        lbl_patch = label[z:z+pd, y:y+ph, x:x+pw]
        
        # Pad if needed
        if img_patch.shape[1:] != self.patch_size:
            img_patch = self._pad_to_size(img_patch, (image.shape[0],) + self.patch_size)
            lbl_patch = self._pad_to_size(lbl_patch, self.patch_size, pad_value=0)
        
        # Relabel connected components - instances may split when cropped
        lbl_patch = relabel_connected_components(lbl_patch)
        
        # Create semantic + instance label
        # semantic: background=0, foreground=semantic_class_id (configurable)
        # e.g., semantic_class_id=1 for neuron, 2 for mitochondria
        semantic = np.where(lbl_patch > 0, self.semantic_class_id, 0).astype(np.float32)
        combined_label = np.stack([semantic, lbl_patch.astype(np.float32)], axis=0)
        
        return {
            'image': torch.from_numpy(img_patch).float(),
            'label': torch.from_numpy(combined_label).float()
        }
    
    def _pad_to_size(
        self,
        arr: np.ndarray,
        target_size: Tuple[int, ...],
        pad_value: float = 0
    ) -> np.ndarray:
        """Pad array to target size."""
        pad_width = []
        for current, target in zip(arr.shape, target_size):
            total_pad = max(0, target - current)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        
        return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    
    def _augment(self, patch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentations using einops for rotations."""
        image = patch['image']
        label = patch['label']
        
        # Random flips along each spatial axis
        for axis in [1, 2, 3]:  # D, H, W (skip channel dim)
            if np.random.random() > 0.5:
                image = torch.flip(image, dims=[axis])
                label = torch.flip(label, dims=[axis])
        
        # Random 90° rotation in XY plane
        k = np.random.randint(0, 4)
        if k > 0:
            # Rotate in H-W plane (axes 2 and 3)
            image = torch.rot90(image, k, dims=[2, 3])
            label = torch.rot90(label, k, dims=[2, 3])
        
        # Intensity augmentation (image only)
        if np.random.random() > 0.5:
            brightness = (np.random.random() - 0.5) * 0.2
            image = torch.clamp(image + brightness, 0, 1)
        
        if np.random.random() > 0.5:
            # Contrast adjustment
            mean_val = reduce(image, 'c d h w -> 1 1 1 1', 'mean')
            contrast = 0.8 + np.random.random() * 0.4
            image = torch.clamp((image - mean_val) * contrast + mean_val, 0, 1)
        
        # Gaussian noise
        if np.random.random() > 0.7:
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
        
        return {'image': image, 'label': label}


class SNEMI3DDataset(UnifiedDataset):
    """
    Specialized dataset for SNEMI3D connectomics benchmark.
    
    SNEMI3D format:
    - {prefix}_inputs.tiff: EM images (100, 1024, 1024) uint8
    - {prefix}_labels.tiff: Instance labels (100, 1024, 1024) uint16
    
    Semantic class: neuron (class_id=1)
    """
    
    def __init__(
        self,
        data_root: str,
        volumes: List[str],  # e.g., ['AC3', 'AC4']
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        patches_per_volume: int = 100,
        augment: bool = True,
        cache_volumes: bool = True,
        semantic_class_id: int = 1  # neuron
    ):
        # Build paths from volume names
        image_paths = [f"{vol}_inputs.tiff" for vol in volumes]
        label_paths = [f"{vol}_labels.tiff" for vol in volumes]
        
        super().__init__(
            data_root=data_root,
            image_paths=image_paths,
            label_paths=label_paths,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume,
            augment=augment,
            cache_volumes=cache_volumes,
            normalize_images=True,
            semantic_class_id=semantic_class_id
        )
        
        self.volumes = volumes


class MitoEM2Dataset(UnifiedDataset):
    """
    Specialized dataset for MitoEM 2.0 mitochondria benchmark.
    
    MitoEM 2.0 format (nnUNet style):
    - imagesTr/{dataset}_{case}_0000.tif: EM images
    - labelsTr/{dataset}_{case}.tif: Instance labels for mitochondria
    
    Semantic class: mitochondria (class_id=2)
    
    Example datasets:
    - Dataset001_ME2-Beta, Dataset002_ME2-Jurkat, etc.
    """
    
    def __init__(
        self,
        data_root: str,
        image_paths: List[str],  # e.g., ['imagesTr/Beta_001_0000.tif']
        label_paths: List[str],  # e.g., ['labelsTr/Beta_001.tif']
        patch_size: Tuple[int, int, int] = (64, 256, 256),
        patches_per_volume: int = 100,
        augment: bool = True,
        cache_volumes: bool = True,
        semantic_class_id: int = 2  # mitochondria
    ):
        super().__init__(
            data_root=data_root,
            image_paths=image_paths,
            label_paths=label_paths,
            patch_size=patch_size,
            patches_per_volume=patches_per_volume,
            augment=augment,
            cache_volumes=cache_volumes,
            normalize_images=True,
            semantic_class_id=semantic_class_id
        )


class SNEMI3DDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for SNEMI3D dataset.
    
    Supports both Hydra instantiation (keyword args) and direct cfg passing.
    """
    
    def __init__(
        self,
        data_root: str = 'data/',
        train_volumes: List[str] = None,
        val_volumes: List[str] = None,
        patch_size: Union[int, List[int], Tuple[int, ...]] = 64,
        patches_per_volume: int = 100,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        cache_volumes: bool = True,
        **kwargs  # Accept extra kwargs from Hydra (like _target_)
    ):
        super().__init__()
        
        self.data_root = data_root
        self.train_volumes = train_volumes if train_volumes is not None else ['AC3']
        self.val_volumes = val_volumes if val_volumes is not None else ['AC4']
        
        # Handle patch_size formats (Hydra passes ListConfig, not list)
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            # Convert any sequence-like (list, tuple, ListConfig) to tuple
            try:
                self.patch_size = tuple(patch_size)
            except (TypeError, ValueError):
                self.patch_size = (64, 128, 128)
        
        print(f"[SNEMI3DDataModule] patch_size received: {patch_size} (type: {type(patch_size).__name__})")
        print(f"[SNEMI3DDataModule] patch_size resolved: {self.patch_size}")
        
        self.patches_per_volume = patches_per_volume
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_volumes = cache_volumes
        
        self.train_ds = None
        self.val_ds = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_ds = SNEMI3DDataset(
                data_root=self.data_root,
                volumes=self.train_volumes,
                patch_size=self.patch_size,
                patches_per_volume=self.patches_per_volume,
                augment=True,
                cache_volumes=self.cache_volumes
            )
            
            if self.val_volumes:
                self.val_ds = SNEMI3DDataset(
                    data_root=self.data_root,
                    volumes=self.val_volumes,
                    patch_size=self.patch_size,
                    patches_per_volume=self.patches_per_volume // 4,
                    augment=False,
                    cache_volumes=self.cache_volumes
                )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class MitoEM2DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MitoEM 2.0 mitochondria dataset.
    
    MitoEM 2.0 provides multiple cell type datasets with mitochondria labels.
    Uses nnUNet-style folder structure:
    - imagesTr/: Training images (*.tif)
    - labelsTr/: Training labels (*.tif)
    
    Semantic class: mitochondria (class_id=2)
    """
    
    def __init__(
        self,
        data_root: str = 'data/MitoEM2/',
        train_image_paths: List[str] = None,
        train_label_paths: List[str] = None,
        val_image_paths: List[str] = None,
        val_label_paths: List[str] = None,
        patch_size: Union[int, List[int], Tuple[int, ...]] = (64, 256, 256),
        patches_per_volume: int = 100,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        cache_volumes: bool = True,
        semantic_class_id: int = 2,  # mitochondria
        **kwargs
    ):
        super().__init__()
        
        self.data_root = data_root
        self.train_image_paths = train_image_paths or []
        self.train_label_paths = train_label_paths or []
        self.val_image_paths = val_image_paths or []
        self.val_label_paths = val_label_paths or []
        
        # Handle patch_size formats (Hydra passes ListConfig, not list)
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            try:
                self.patch_size = tuple(patch_size)
            except (TypeError, ValueError):
                self.patch_size = (64, 256, 256)
        
        print(f"[MitoEM2DataModule] patch_size resolved: {self.patch_size}")
        
        self.patches_per_volume = patches_per_volume
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_volumes = cache_volumes
        self.semantic_class_id = semantic_class_id
        
        self.train_ds = None
        self.val_ds = None
    
    @classmethod
    def from_dataset_folder(
        cls,
        dataset_folder: str,
        train_ratio: float = 0.8,
        **kwargs
    ) -> 'MitoEM2DataModule':
        """
        Create DataModule from nnUNet-style dataset folder.
        
        Args:
            dataset_folder: Path to dataset (e.g., 'data/Dataset001_ME2-Beta/')
            train_ratio: Fraction of data for training
        """
        import glob
        
        images_dir = Path(dataset_folder) / 'imagesTr'
        labels_dir = Path(dataset_folder) / 'labelsTr'
        
        # Find all image files
        image_files = sorted(glob.glob(str(images_dir / '*.tif')))
        
        # Build corresponding label paths
        image_paths = []
        label_paths = []
        for img_path in image_files:
            img_name = Path(img_path).name
            # Convert image name to label name (remove _0000 suffix)
            lbl_name = img_name.replace('_0000.tif', '.tif')
            lbl_path = labels_dir / lbl_name
            
            if lbl_path.exists():
                # Store relative paths
                image_paths.append(str(Path('imagesTr') / img_name))
                label_paths.append(str(Path('labelsTr') / lbl_name))
        
        # Split into train/val
        n_train = int(len(image_paths) * train_ratio)
        
        return cls(
            data_root=dataset_folder,
            train_image_paths=image_paths[:n_train],
            train_label_paths=label_paths[:n_train],
            val_image_paths=image_paths[n_train:],
            val_label_paths=label_paths[n_train:],
            **kwargs
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            if self.train_image_paths:
                self.train_ds = MitoEM2Dataset(
                    data_root=self.data_root,
                    image_paths=self.train_image_paths,
                    label_paths=self.train_label_paths,
                    patch_size=self.patch_size,
                    patches_per_volume=self.patches_per_volume,
                    augment=True,
                    cache_volumes=self.cache_volumes,
                    semantic_class_id=self.semantic_class_id
                )
            
            if self.val_image_paths:
                self.val_ds = MitoEM2Dataset(
                    data_root=self.data_root,
                    image_paths=self.val_image_paths,
                    label_paths=self.val_label_paths,
                    patch_size=self.patch_size,
                    patches_per_volume=self.patches_per_volume // 4,
                    augment=False,
                    cache_volumes=self.cache_volumes,
                    semantic_class_id=self.semantic_class_id
                )
    
    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("No training data configured")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class ConnectomicsDataModule(pl.LightningDataModule):
    """
    Generic DataModule supporting multiple data formats.
    
    Uses JSON data list for flexible dataset configuration.
    Supports both Hydra instantiation (keyword args) and direct usage.
    """
    
    def __init__(
        self,
        data_list_path: str = 'data/dataset.json',
        data_root: str = 'data/',
        patch_size: Union[int, List[int], Tuple[int, ...]] = 128,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        cache_rate: float = 1.0,
        image_key: str = 'image',
        label_key: str = 'label',
        h5_image_key: str = 'raw',
        h5_label_key: str = 'label',
        augmentation: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept extra kwargs from Hydra
    ):
        super().__init__()
        
        self.data_list_path = data_list_path
        self.data_root = data_root
        
        # Handle patch_size formats (Hydra passes ListConfig, not list)
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            try:
                self.patch_size = tuple(patch_size)
            except (TypeError, ValueError):
                self.patch_size = (128, 128, 128)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_rate = cache_rate
        
        self.image_key = image_key
        self.label_key = label_key
        self.keys = [self.image_key, self.label_key]
        
        # HDF5 specific settings
        self.h5_image_key = h5_image_key
        self.h5_label_key = h5_label_key
        
        # Augmentation config
        self.augmentation = augmentation or {}
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
    
    def setup(self, stage: Optional[str] = None):
        data_list = self._load_data_list()
        
        train_transforms = self._build_train_transforms()
        val_transforms = self._build_val_transforms()
        
        if stage == 'fit' or stage is None:
            train_data = data_list.get('training', [])
            val_data = data_list.get('validation', [])
            
            if train_data:
                self.train_ds = CacheDataset(
                    data=train_data,
                    transform=train_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers
                )
            
            if val_data:
                self.val_ds = CacheDataset(
                    data=val_data,
                    transform=val_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers
                )
        
        if stage == 'test' or stage is None:
            test_data = data_list.get('test', [])
            if test_data:
                self.test_ds = CacheDataset(
                    data=test_data,
                    transform=val_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers
                )
    
    def _load_data_list(self) -> Dict[str, Any]:
        if not os.path.exists(self.data_list_path):
            return {'training': [], 'validation': [], 'test': []}
        
        with open(self.data_list_path, 'r') as f:
            data_list = json.load(f)
        
        for split in ['training', 'validation', 'test']:
            if split in data_list:
                for item in data_list[split]:
                    if self.image_key in item and not os.path.isabs(item[self.image_key]):
                        item[self.image_key] = os.path.join(self.data_root, item[self.image_key])
                    if self.label_key in item and not os.path.isabs(item[self.label_key]):
                        item[self.label_key] = os.path.join(self.data_root, item[self.label_key])
        
        return data_list
    
    def _build_train_transforms(self) -> Compose:
        aug_cfg = self.augmentation
        
        transforms = [
            LoadWithReaderd(keys=self.keys, h5_dataset_key=self.h5_image_key),
            EnsureChannelFirstd(keys=self.keys),
            ScaleIntensityd(keys=[self.image_key]),
            SpatialPadd(keys=self.keys, spatial_size=self.patch_size),
            RandCropByPosNegLabeld(
                keys=self.keys,
                label_key=self.label_key,
                spatial_size=self.patch_size,
                pos=1, neg=1,
                num_samples=4,
                image_key=self.image_key,
                image_threshold=0
            ),
        ]
        
        if aug_cfg.get('enabled', True):
            if aug_cfg.get('random_rotate_prob', 0) > 0:
                transforms.append(
                    RandRotate90d(
                        keys=self.keys,
                        prob=aug_cfg.get('random_rotate_prob', 0.5),
                        max_k=3, spatial_axes=(0, 1)
                    )
                )
            
            if aug_cfg.get('random_flip_prob', 0) > 0:
                for axis in [0, 1, 2]:
                    transforms.append(
                        RandFlipd(
                            keys=self.keys,
                            prob=aug_cfg.get('random_flip_prob', 0.5),
                            spatial_axis=axis
                        )
                    )
            
            transforms.append(
                RandGaussianNoised(keys=[self.image_key], prob=0.3, mean=0.0, std=0.1)
            )
        
        transforms.append(EnsureTyped(keys=self.keys))
        return Compose(transforms)
    
    def _build_val_transforms(self) -> Compose:
        return Compose([
            LoadWithReaderd(keys=self.keys, h5_dataset_key=self.h5_image_key),
            EnsureChannelFirstd(keys=self.keys),
            ScaleIntensityd(keys=[self.image_key]),
            SpatialPadd(keys=self.keys, spatial_size=self.patch_size),
            RandCropByPosNegLabeld(
                keys=self.keys,
                label_key=self.label_key,
                spatial_size=self.patch_size,
                pos=1, neg=0, num_samples=1,
                image_key=self.image_key,
                image_threshold=0
            ),
            EnsureTyped(keys=self.keys)
        ])
    
    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Training dataset not initialized.")
        
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.pin_memory
        )


class InferenceDataModule(pl.LightningDataModule):
    """
    DataModule for inference using sliding window.
    
    Supports all formats via AutoReader.
    """
    
    def __init__(
        self,
        volume_path: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        overlap: float = 0.5,
        batch_size: int = 4,
        num_workers: int = 4,
        h5_dataset_key: str = 'raw'
    ):
        super().__init__()
        self.volume_path = volume_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.h5_dataset_key = h5_dataset_key
    
    def setup(self, stage: Optional[str] = None):
        reader = AutoReader(
            normalize=True,
            add_channel=True,
            dataset_key=self.h5_dataset_key
        )
        
        volume = reader.read(self.volume_path)
        
        self.predict_ds = VolumeSliceDataset(
            volume=volume,
            patch_size=self.patch_size,
            overlap=self.overlap
        )
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class VolumeSliceDataset(TorchDataset):
    """
    Dataset for sliding window inference.
    
    Uses einops for tensor manipulation.
    """
    
    def __init__(
        self,
        volume: np.ndarray,
        patch_size: Tuple[int, int, int],
        overlap: float = 0.5
    ):
        self.volume = torch.from_numpy(volume).float()
        self.patch_size = patch_size
        self.stride = tuple(int(s * (1 - overlap)) for s in patch_size)
        self.positions = self._calculate_positions()
    
    def _calculate_positions(self) -> List[Tuple[int, int, int]]:
        _, d, h, w = self.volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        
        positions = []
        for z in range(0, max(1, d - pd + 1), sd):
            for y in range(0, max(1, h - ph + 1), sh):
                for x in range(0, max(1, w - pw + 1), sw):
                    positions.append((z, y, x))
        
        return positions
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        z, y, x = self.positions[idx]
        pd, ph, pw = self.patch_size
        
        patch = self.volume[:, z:z+pd, y:y+ph, x:x+pw]
        
        return {
            'image': patch,
            'coordinates': torch.tensor([z, y, x]),
            'patch_size': torch.tensor(self.patch_size)
        }


class CombinedDataModule(pl.LightningDataModule):
    """
    Combined DataModule for multiple connectomics datasets.
    
    Supports combining SNEMI3D (neurons) and MitoEM2 (mitochondria) datasets
    into a single training pipeline.
    
    Args:
        datasets: List of dataset configurations, each containing:
            - type: 'snemi3d' or 'mitoem2'
            - data_root: Path to dataset
            - volumes/datasets: Volume names or dataset folders
            - semantic_class_id: Class ID for this dataset
        patch_size: Patch size (D, H, W)
        patches_per_volume: Patches per volume per epoch
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        cache_volumes: Cache volumes in memory
    """
    
    def __init__(
        self,
        datasets: List[Dict[str, Any]] = None,
        patch_size: Union[int, List[int], Tuple[int, ...]] = (32, 256, 256),
        patches_per_volume: int = 100,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        cache_volumes: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.dataset_configs = datasets or []
        
        # Handle patch_size formats
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            try:
                self.patch_size = tuple(patch_size)
            except (TypeError, ValueError):
                self.patch_size = (32, 256, 256)
        
        self.patches_per_volume = patches_per_volume
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_volumes = cache_volumes
        
        self.train_ds = None
        self.val_ds = None
        
        print(f"[CombinedDataModule] patch_size: {self.patch_size}")
        print(f"[CombinedDataModule] datasets: {len(self.dataset_configs)}")
    
    def _create_snemi3d_dataset(
        self,
        config: Dict[str, Any],
        is_train: bool = True
    ) -> TorchDataset:
        """Create SNEMI3D dataset from config."""
        data_root = config.get('data_root', 'data/SNEMI3D/')
        volumes = config.get('train_volumes' if is_train else 'val_volumes', [])
        semantic_class_id = config.get('semantic_class_id', 1)  # neuron
        
        if not volumes:
            return None
        
        return SNEMI3DDataset(
            data_root=data_root,
            volumes=volumes,
            patch_size=self.patch_size,
            patches_per_volume=self.patches_per_volume if is_train else self.patches_per_volume // 4,
            augment=is_train,
            cache_volumes=self.cache_volumes,
            semantic_class_id=semantic_class_id
        )
    
    def _create_mitoem2_dataset(
        self,
        config: Dict[str, Any],
        is_train: bool = True
    ) -> TorchDataset:
        """Create MitoEM2 dataset from config."""
        import glob
        
        data_root = config.get('data_root', 'data/MitoEM2/')
        dataset_folders = config.get('datasets', [])
        semantic_class_id = config.get('semantic_class_id', 2)  # mitochondria
        train_ratio = config.get('train_ratio', 0.8)
        
        if not dataset_folders:
            return None
        
        all_image_paths = []
        all_label_paths = []
        
        for folder in dataset_folders:
            folder_path = Path(data_root) / folder
            images_dir = folder_path / 'imagesTr'
            labels_dir = folder_path / 'labelsTr'
            
            # Support both .nii.gz and .tif files
            image_files = sorted(
                glob.glob(str(images_dir / '*_0000.nii.gz')) +
                glob.glob(str(images_dir / '*_0000.tif'))
            )
            
            for img_path in image_files:
                img_name = Path(img_path).name
                # Convert image name to label name (remove _0000 suffix)
                if img_name.endswith('_0000.nii.gz'):
                    lbl_name = img_name.replace('_0000.nii.gz', '.nii.gz')
                else:
                    lbl_name = img_name.replace('_0000.tif', '.tif')
                
                lbl_path = labels_dir / lbl_name
                
                if lbl_path.exists():
                    # Store paths relative to data_root
                    rel_img = str(Path(folder) / 'imagesTr' / img_name)
                    rel_lbl = str(Path(folder) / 'labelsTr' / lbl_name)
                    all_image_paths.append(rel_img)
                    all_label_paths.append(rel_lbl)
        
        if not all_image_paths:
            return None
        
        # Split train/val
        n_train = int(len(all_image_paths) * train_ratio)
        
        if is_train:
            image_paths = all_image_paths[:n_train]
            label_paths = all_label_paths[:n_train]
        else:
            image_paths = all_image_paths[n_train:]
            label_paths = all_label_paths[n_train:]
        
        if not image_paths:
            return None
        
        return MitoEM2Dataset(
            data_root=data_root,
            image_paths=image_paths,
            label_paths=label_paths,
            patch_size=self.patch_size,
            patches_per_volume=self.patches_per_volume if is_train else self.patches_per_volume // 4,
            augment=is_train,
            cache_volumes=self.cache_volumes,
            semantic_class_id=semantic_class_id
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup combined datasets."""
        if stage == 'fit' or stage is None:
            train_datasets = []
            val_datasets = []
            
            for config in self.dataset_configs:
                ds_type = config.get('type', 'snemi3d').lower()
                
                if ds_type == 'snemi3d':
                    train_ds = self._create_snemi3d_dataset(config, is_train=True)
                    val_ds = self._create_snemi3d_dataset(config, is_train=False)
                elif ds_type == 'mitoem2':
                    train_ds = self._create_mitoem2_dataset(config, is_train=True)
                    val_ds = self._create_mitoem2_dataset(config, is_train=False)
                else:
                    print(f"Warning: Unknown dataset type: {ds_type}")
                    continue
                
                if train_ds is not None:
                    train_datasets.append(train_ds)
                    print(f"  Added {ds_type} train: {len(train_ds)} samples")
                if val_ds is not None:
                    val_datasets.append(val_ds)
                    print(f"  Added {ds_type} val: {len(val_ds)} samples")
            
            # Combine datasets
            if train_datasets:
                self.train_ds = torch.utils.data.ConcatDataset(train_datasets)
                print(f"[CombinedDataModule] Total train samples: {len(self.train_ds)}")
            
            if val_datasets:
                self.val_ds = torch.utils.data.ConcatDataset(val_datasets)
                print(f"[CombinedDataModule] Total val samples: {len(self.val_ds)}")
    
    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("No training data configured")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
