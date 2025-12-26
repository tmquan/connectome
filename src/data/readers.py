"""
Data Readers for Connectomics Volumes

Individual readers for different volumetric data formats:
- TIFF: Standard electron microscopy format
- HDF5/H5: Hierarchical data format for large datasets
- NIfTI: Neuroimaging format (.nii, .nii.gz)

Each reader provides a consistent interface for loading volumes
and their associated metadata.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from einops import rearrange, repeat

# Optional imports with graceful fallbacks
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class VolumeReader(ABC):
    """
    Abstract base class for volume readers.
    
    All readers must implement the read() method and provide
    consistent output format: (C, D, H, W) for images.
    """
    
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def __init__(self, normalize: bool = True, add_channel: bool = True):
        """
        Args:
            normalize: Whether to normalize values to [0, 1]
            add_channel: Whether to add channel dimension if missing
        """
        self.normalize = normalize
        self.add_channel = add_channel
    
    @abstractmethod
    def read(self, path: str) -> np.ndarray:
        """
        Read volume from file.
        
        Args:
            path: Path to the volume file
            
        Returns:
            Volume array in (C, D, H, W) format
        """
        pass
    
    @abstractmethod
    def read_with_metadata(self, path: str) -> Dict[str, Any]:
        """
        Read volume with associated metadata.
        
        Args:
            path: Path to the volume file
            
        Returns:
            Dictionary containing 'data' and 'metadata' keys
        """
        pass
    
    def _ensure_4d(self, volume: np.ndarray) -> np.ndarray:
        """Ensure volume has 4 dimensions (C, D, H, W) if add_channel is True."""
        if volume.ndim == 3:
            if self.add_channel:
                # (D, H, W) -> (1, D, H, W)
                volume = rearrange(volume, 'd h w -> 1 d h w')
            # else: keep as 3D
        elif volume.ndim == 4:
            # Already 4D
            pass
        elif volume.ndim == 2:
            # 2D image - add depth and optionally channel
            if self.add_channel:
                volume = rearrange(volume, 'h w -> 1 1 h w')
            else:
                volume = rearrange(volume, 'h w -> 1 h w')
        # For other dimensions, just return as-is
        return volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range."""
        if not self.normalize:
            return volume
        
        volume = volume.astype(np.float32)
        vmin, vmax = volume.min(), volume.max()
        
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
        
        return volume
    
    @classmethod
    def supports_file(cls, path: str) -> bool:
        """Check if this reader supports the given file."""
        ext = Path(path).suffix.lower()
        # Handle .nii.gz
        if path.lower().endswith('.nii.gz'):
            ext = '.nii.gz'
        return ext in cls.SUPPORTED_EXTENSIONS


class TiffReader(VolumeReader):
    """
    Reader for TIFF/TIF volumetric images.
    
    Common format for electron microscopy data including:
    - SNEMI3D dataset
    - CREMI dataset
    - ISBI challenges
    """
    
    SUPPORTED_EXTENSIONS = ['.tiff', '.tif']
    
    def __init__(
        self, 
        normalize: bool = True, 
        add_channel: bool = True,
        series: int = 0
    ):
        """
        Args:
            normalize: Normalize to [0, 1]
            add_channel: Add channel dimension
            series: Which series to read for multi-series TIFF
        """
        super().__init__(normalize, add_channel)
        
        if not HAS_TIFFFILE:
            raise ImportError(
                "tifffile is required for TIFF support. "
                "Install with: pip install tifffile"
            )
        
        self.series = series
    
    def read(self, path: str) -> np.ndarray:
        """Read TIFF volume."""
        volume = tifffile.imread(path)
        volume = volume.astype(np.float32)
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        volume = self._ensure_4d(volume)
        
        return volume
    
    def read_with_metadata(self, path: str) -> Dict[str, Any]:
        """Read TIFF with metadata."""
        with tifffile.TiffFile(path) as tif:
            volume = tif.asarray()
            
            metadata = {
                'shape': volume.shape,
                'dtype': str(volume.dtype),
                'pages': len(tif.pages),
                'is_bigtiff': tif.is_bigtiff,
            }
            
            # Extract resolution if available
            if tif.pages[0].tags.get('XResolution'):
                metadata['x_resolution'] = tif.pages[0].tags['XResolution'].value
            if tif.pages[0].tags.get('YResolution'):
                metadata['y_resolution'] = tif.pages[0].tags['YResolution'].value
            
            # ImageJ metadata if present
            if tif.imagej_metadata:
                metadata['imagej'] = tif.imagej_metadata
        
        volume = volume.astype(np.float32)
        if self.normalize:
            volume = self._normalize_volume(volume)
        volume = self._ensure_4d(volume)
        
        return {'data': volume, 'metadata': metadata}
    
    def write(
        self, 
        path: str, 
        volume: np.ndarray,
        compress: int = 0,
        dtype: np.dtype = np.uint8
    ):
        """
        Write volume to TIFF file.
        
        Args:
            path: Output path
            volume: Volume array (C, D, H, W) or (D, H, W)
            compress: Compression level (0=none, 1-9)
            dtype: Output data type
        """
        # Remove channel dimension if single channel
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = rearrange(volume, '1 d h w -> d h w')
        
        # Scale to dtype range
        if dtype == np.uint8:
            volume = (volume * 255).clip(0, 255).astype(np.uint8)
        elif dtype == np.uint16:
            volume = (volume * 65535).clip(0, 65535).astype(np.uint16)
        else:
            volume = volume.astype(dtype)
        
        tifffile.imwrite(path, volume, compress=compress)


class H5Reader(VolumeReader):
    """
    Reader for HDF5/H5 volumetric data.
    
    Supports hierarchical datasets common in:
    - Large-scale connectomics (FIBSEM, etc.)
    - Multi-resolution pyramids
    - Chunked datasets for out-of-core processing
    """
    
    SUPPORTED_EXTENSIONS = ['.h5', '.hdf5', '.hdf', '.he5']
    
    def __init__(
        self,
        normalize: bool = True,
        add_channel: bool = True,
        dataset_key: str = 'data',
        fallback_keys: Optional[List[str]] = None
    ):
        """
        Args:
            normalize: Normalize to [0, 1]
            add_channel: Add channel dimension
            dataset_key: Primary HDF5 dataset key to read
            fallback_keys: Alternative keys to try if primary not found
        """
        super().__init__(normalize, add_channel)
        
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 support. "
                "Install with: pip install h5py"
            )
        
        self.dataset_key = dataset_key
        self.fallback_keys = fallback_keys or [
            'raw', 'volume', 'image', 'em', 'main',
            'volumes/raw', 'images/raw', 'data/raw'
        ]
    
    def _find_dataset_key(self, h5file: 'h5py.File') -> str:
        """Find the correct dataset key in the file."""
        # Try primary key
        if self.dataset_key in h5file:
            return self.dataset_key
        
        # Try fallback keys
        for key in self.fallback_keys:
            if key in h5file:
                return key
        
        # List available keys
        available = list(h5file.keys())
        raise KeyError(
            f"Dataset key '{self.dataset_key}' not found. "
            f"Available keys: {available}"
        )
    
    def read(self, path: str) -> np.ndarray:
        """Read HDF5 volume."""
        with h5py.File(path, 'r') as f:
            key = self._find_dataset_key(f)
            volume = f[key][:]
        
        volume = volume.astype(np.float32)
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        volume = self._ensure_4d(volume)
        
        return volume
    
    def read_with_metadata(self, path: str) -> Dict[str, Any]:
        """Read HDF5 with metadata and attributes."""
        with h5py.File(path, 'r') as f:
            key = self._find_dataset_key(f)
            dataset = f[key]
            volume = dataset[:]
            
            metadata = {
                'shape': volume.shape,
                'dtype': str(volume.dtype),
                'dataset_key': key,
                'chunks': dataset.chunks,
                'compression': dataset.compression,
            }
            
            # Read dataset attributes
            metadata['attributes'] = dict(dataset.attrs)
            
            # Read file-level attributes
            metadata['file_attributes'] = dict(f.attrs)
            
            # List all groups and datasets
            metadata['structure'] = self._get_h5_structure(f)
        
        volume = volume.astype(np.float32)
        if self.normalize:
            volume = self._normalize_volume(volume)
        volume = self._ensure_4d(volume)
        
        return {'data': volume, 'metadata': metadata}
    
    def _get_h5_structure(self, h5file: 'h5py.File', prefix: str = '') -> Dict:
        """Recursively get HDF5 file structure."""
        structure = {}
        for key in h5file.keys():
            full_key = f"{prefix}/{key}" if prefix else key
            item = h5file[key]
            if isinstance(item, h5py.Dataset):
                structure[key] = {
                    'type': 'dataset',
                    'shape': item.shape,
                    'dtype': str(item.dtype)
                }
            elif isinstance(item, h5py.Group):
                structure[key] = {
                    'type': 'group',
                    'contents': self._get_h5_structure(item, full_key)
                }
        return structure
    
    def read_roi(
        self,
        path: str,
        start: Tuple[int, ...],
        size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Read a region of interest from HDF5 (useful for large files).
        
        Args:
            path: File path
            start: Starting coordinates (d, h, w) or (c, d, h, w)
            size: Size of ROI
            
        Returns:
            ROI array
        """
        with h5py.File(path, 'r') as f:
            key = self._find_dataset_key(f)
            dataset = f[key]
            
            # Build slice
            slices = tuple(slice(s, s + sz) for s, sz in zip(start, size))
            volume = dataset[slices]
        
        volume = volume.astype(np.float32)
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        return volume
    
    def write(
        self,
        path: str,
        volume: np.ndarray,
        dataset_key: str = 'data',
        chunks: Optional[Tuple[int, ...]] = None,
        compression: str = 'gzip',
        compression_opts: int = 4
    ):
        """
        Write volume to HDF5 file.
        
        Args:
            path: Output path
            volume: Volume array
            dataset_key: Dataset name in HDF5 file
            chunks: Chunk shape for storage
            compression: Compression algorithm
            compression_opts: Compression level
        """
        with h5py.File(path, 'w') as f:
            f.create_dataset(
                dataset_key,
                data=volume,
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts
            )


class NiftiReader(VolumeReader):
    """
    Reader for NIfTI neuroimaging format.
    
    Supports:
    - .nii files
    - .nii.gz compressed files
    - Affine transformation metadata
    - Voxel spacing information
    """
    
    SUPPORTED_EXTENSIONS = ['.nii', '.nii.gz']
    
    def __init__(
        self,
        normalize: bool = True,
        add_channel: bool = True,
        reorient: bool = False,
        target_orientation: str = 'RAS'
    ):
        """
        Args:
            normalize: Normalize to [0, 1]
            add_channel: Add channel dimension
            reorient: Whether to reorient to standard orientation
            target_orientation: Target orientation (e.g., 'RAS', 'LPS')
        """
        super().__init__(normalize, add_channel)
        
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required for NIfTI support. "
                "Install with: pip install nibabel"
            )
        
        self.reorient = reorient
        self.target_orientation = target_orientation
    
    def read(self, path: str) -> np.ndarray:
        """Read NIfTI volume."""
        img = nib.load(path)
        
        if self.reorient:
            img = nib.as_closest_canonical(img)
        
        volume = img.get_fdata().astype(np.float32)
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        volume = self._ensure_4d(volume)
        
        return volume
    
    def read_with_metadata(self, path: str) -> Dict[str, Any]:
        """Read NIfTI with full header metadata."""
        img = nib.load(path)
        
        if self.reorient:
            img = nib.as_closest_canonical(img)
        
        volume = img.get_fdata().astype(np.float32)
        header = img.header
        
        metadata = {
            'shape': volume.shape,
            'dtype': str(volume.dtype),
            'affine': img.affine.tolist(),
            'voxel_sizes': header.get_zooms(),
            'dimensions': header.get_data_shape(),
            'units': (header.get_xyzt_units()[0], header.get_xyzt_units()[1]),
            'orientation': nib.aff2axcodes(img.affine),
        }
        
        # Additional header fields
        if hasattr(header, 'get_qform'):
            metadata['qform'] = header.get_qform().tolist()
        if hasattr(header, 'get_sform'):
            metadata['sform'] = header.get_sform().tolist()
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        volume = self._ensure_4d(volume)
        
        return {'data': volume, 'metadata': metadata}
    
    def write(
        self,
        path: str,
        volume: np.ndarray,
        affine: Optional[np.ndarray] = None,
        header: Optional['nib.Nifti1Header'] = None
    ):
        """
        Write volume to NIfTI file.
        
        Args:
            path: Output path
            volume: Volume array (C, D, H, W) or (D, H, W)
            affine: Affine transformation matrix
            header: NIfTI header
        """
        # Remove channel dimension if single channel
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = rearrange(volume, '1 d h w -> d h w')
        
        if affine is None:
            affine = np.eye(4)
        
        img = nib.Nifti1Image(volume, affine, header)
        nib.save(img, path)


class AutoReader:
    """
    Automatic format detection and reader selection.
    
    Detects file format from extension and uses appropriate reader.
    """
    
    READERS = {
        'tiff': TiffReader,
        'h5': H5Reader,
        'nifti': NiftiReader
    }
    
    def __init__(self, normalize: bool = True, add_channel: bool = True, **kwargs):
        """
        Args:
            normalize: Normalize values to [0, 1]
            add_channel: Add channel dimension if missing
            **kwargs: Format-specific options
        """
        self.normalize = normalize
        self.add_channel = add_channel
        self.kwargs = kwargs
        
        self._reader_cache: Dict[str, VolumeReader] = {}
    
    def _get_reader(self, path: str) -> VolumeReader:
        """Get appropriate reader for file type."""
        ext = Path(path).suffix.lower()
        
        # Handle .nii.gz
        if path.lower().endswith('.nii.gz'):
            ext = '.nii.gz'
        
        # Determine reader type
        if ext in TiffReader.SUPPORTED_EXTENSIONS:
            reader_type = 'tiff'
        elif ext in H5Reader.SUPPORTED_EXTENSIONS:
            reader_type = 'h5'
        elif ext in NiftiReader.SUPPORTED_EXTENSIONS:
            reader_type = 'nifti'
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Use cached reader if available
        if reader_type not in self._reader_cache:
            reader_class = self.READERS[reader_type]
            
            # Filter kwargs for this reader
            reader_kwargs = {
                'normalize': self.normalize,
                'add_channel': self.add_channel
            }
            
            # Add format-specific kwargs
            if reader_type == 'h5' and 'dataset_key' in self.kwargs:
                reader_kwargs['dataset_key'] = self.kwargs['dataset_key']
            
            self._reader_cache[reader_type] = reader_class(**reader_kwargs)
        
        return self._reader_cache[reader_type]
    
    def read(self, path: str) -> np.ndarray:
        """Read volume with automatic format detection."""
        reader = self._get_reader(path)
        return reader.read(path)
    
    def read_with_metadata(self, path: str) -> Dict[str, Any]:
        """Read volume with metadata and automatic format detection."""
        reader = self._get_reader(path)
        return reader.read_with_metadata(path)


def get_reader(
    format: Optional[str] = None,
    **kwargs
) -> Union[VolumeReader, AutoReader]:
    """
    Factory function to get appropriate reader.
    
    Args:
        format: Explicit format ('tiff', 'h5', 'nifti') or None for auto
        **kwargs: Reader configuration options
        
    Returns:
        Appropriate reader instance
    """
    if format is None:
        return AutoReader(**kwargs)
    
    format = format.lower()
    
    if format in ['tiff', 'tif']:
        return TiffReader(**kwargs)
    elif format in ['h5', 'hdf5', 'hdf']:
        return H5Reader(**kwargs)
    elif format in ['nifti', 'nii']:
        return NiftiReader(**kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")


# Convenience functions
def read_volume(
    path: str,
    normalize: bool = True,
    add_channel: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Convenience function to read any supported volume format.
    
    Args:
        path: Path to volume file
        normalize: Normalize to [0, 1]
        add_channel: Add channel dimension
        **kwargs: Format-specific options
        
    Returns:
        Volume array in (C, D, H, W) format
    """
    reader = AutoReader(normalize=normalize, add_channel=add_channel, **kwargs)
    return reader.read(path)


def read_volume_with_metadata(
    path: str,
    normalize: bool = True,
    add_channel: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to read volume with metadata.
    
    Args:
        path: Path to volume file
        normalize: Normalize to [0, 1]
        add_channel: Add channel dimension
        **kwargs: Format-specific options
        
    Returns:
        Dictionary with 'data' and 'metadata' keys
    """
    reader = AutoReader(normalize=normalize, add_channel=add_channel, **kwargs)
    return reader.read_with_metadata(path)

