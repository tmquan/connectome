"""
Label Registry Utilities

Provides the interface between human-readable connectomics ontology
and the VISTA3D model's tensor inputs. Maps biological entity names
to unique integer IDs for embedding lookup.

The registry follows a configuration-driven approach where labels
are defined in Hydra YAML files and loaded at runtime.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import json
import yaml


class ConnectomicsLabels(IntEnum):
    """
    Standard label enumeration for connectomics segmentation.
    
    These IDs correspond to the indices in the VISTA3D embedding
    lookup table and semantic segmentation output channels.
    """
    BACKGROUND = 0
    NEURON = 1
    MITOCHONDRIA = 2
    MEMBRANE = 3
    SYNAPSE = 4


@dataclass
class LabelInfo:
    """Information about a single label class."""
    id: int
    name: str
    description: str = ""
    color: Tuple[int, int, int] = (128, 128, 128)
    is_instance: bool = False  # True for classes requiring instance separation
    parent: Optional[str] = None  # Hierarchical label structure


@dataclass
class LabelRegistry:
    """
    Central registry for managing connectomics label mappings.
    
    Acts as the interface between the human-readable biological ontology
    and the model's internal integer representations.
    
    Attributes:
        labels: Dictionary mapping label names to LabelInfo
        _name_to_id: Cached name to ID mapping
        _id_to_name: Cached ID to name mapping
    """
    labels: Dict[str, LabelInfo] = field(default_factory=dict)
    _name_to_id: Dict[str, int] = field(default_factory=dict, repr=False)
    _id_to_name: Dict[int, str] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Build lookup caches after initialization."""
        self._rebuild_caches()
    
    def _rebuild_caches(self):
        """Rebuild name-to-id and id-to-name lookup caches."""
        self._name_to_id = {name: info.id for name, info in self.labels.items()}
        self._id_to_name = {info.id: name for name, info in self.labels.items()}
    
    def add_label(
        self,
        name: str,
        id: int,
        description: str = "",
        color: Tuple[int, int, int] = (128, 128, 128),
        is_instance: bool = False,
        parent: Optional[str] = None
    ):
        """Add a new label to the registry."""
        self.labels[name] = LabelInfo(
            id=id,
            name=name,
            description=description,
            color=color,
            is_instance=is_instance,
            parent=parent
        )
        self._rebuild_caches()
    
    def name_to_id(self, name: str) -> int:
        """
        Map a label name to its integer ID.
        
        This is the core operation for VISTA3D's integer-to-embedding mechanism:
        f_map: D -> {1, ..., N}
        
        Args:
            name: Label name (e.g., "neuron", "mitochondria")
            
        Returns:
            Integer ID for embedding lookup
            
        Raises:
            KeyError: If label name not in registry
        """
        if name not in self._name_to_id:
            raise KeyError(f"Label '{name}' not found in registry. "
                         f"Available labels: {list(self._name_to_id.keys())}")
        return self._name_to_id[name]
    
    def id_to_name(self, id: int) -> str:
        """Map an integer ID back to label name."""
        if id not in self._id_to_name:
            raise KeyError(f"ID {id} not found in registry")
        return self._id_to_name[id]
    
    def get_instance_classes(self) -> List[str]:
        """Get list of class names requiring instance segmentation."""
        return [name for name, info in self.labels.items() if info.is_instance]
    
    def get_semantic_classes(self) -> List[str]:
        """Get list of class names for semantic segmentation."""
        return [name for name, info in self.labels.items() 
                if not info.is_instance and name != 'background']
    
    def to_dict(self) -> Dict[str, int]:
        """Export as simple name-to-id dictionary for Hydra configs."""
        return self._name_to_id.copy()
    
    @classmethod
    def from_dict(cls, config: Dict[str, int]) -> 'LabelRegistry':
        """
        Create registry from a simple dictionary configuration.
        
        Args:
            config: Dictionary mapping label names to integer IDs
                   e.g., {"background": 0, "neuron": 1, ...}
        """
        registry = cls()
        for name, id in config.items():
            is_instance = name == 'neuron'  # Default assumption
            registry.add_label(name=name, id=id, is_instance=is_instance)
        return registry
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LabelRegistry':
        """Load registry from a YAML configuration file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        registry = cls()
        
        labels_config = config.get('nlp_labels', config)
        for name, value in labels_config.items():
            if isinstance(value, int):
                registry.add_label(name=name, id=value)
            elif isinstance(value, dict):
                registry.add_label(
                    name=name,
                    id=value['id'],
                    description=value.get('description', ''),
                    color=tuple(value.get('color', [128, 128, 128])),
                    is_instance=value.get('is_instance', False)
                )
        
        return registry
    
    @classmethod
    def default(cls) -> 'LabelRegistry':
        """
        Create the default connectomics label registry.
        
        Returns a registry with standard biological entity mappings:
        - background (0): Glial cells, extracellular space
        - neuron (1): Individual neurites/somas (instance class)
        - mitochondria (2): Organelles within neurons
        - membrane (3): Cell boundaries
        - synapse (4): Pre/post-synaptic density
        """
        registry = cls()
        
        registry.add_label(
            name='background',
            id=0,
            description='Glial cells and extracellular space',
            color=(0, 0, 0),
            is_instance=False
        )
        
        registry.add_label(
            name='neuron',
            id=1,
            description='Individual neurites and neuronal somas',
            color=(255, 0, 0),
            is_instance=True  # Primary instance class
        )
        
        registry.add_label(
            name='mitochondria',
            id=2,
            description='Mitochondrial organelles within neurons',
            color=(0, 255, 0),
            is_instance=False,
            parent='neuron'
        )
        
        registry.add_label(
            name='membrane',
            id=3,
            description='Cell membrane boundaries',
            color=(0, 0, 255),
            is_instance=False
        )
        
        registry.add_label(
            name='synapse',
            id=4,
            description='Synaptic structures (pre/post-synaptic density)',
            color=(255, 255, 0),
            is_instance=False,
            parent='neuron'
        )
        
        return registry
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __contains__(self, item: str) -> bool:
        return item in self.labels
    
    def __getitem__(self, key: str) -> int:
        return self.name_to_id(key)


def create_embedding_init_strategy(
    registry: LabelRegistry,
    embedding_dim: int,
    strategy: str = 'normal'
) -> Dict[str, Any]:
    """
    Create initialization strategy for class embeddings.
    
    VISTA3D uses learnable embeddings instead of frozen CLIP vectors.
    This function provides different initialization strategies:
    
    - 'normal': Standard normal N(0, 0.02)
    - 'orthogonal': Orthogonal initialization
    - 'zero': Zero initialization (not recommended)
    
    Args:
        registry: Label registry with class definitions
        embedding_dim: Dimension of embedding vectors
        strategy: Initialization strategy name
        
    Returns:
        Dictionary with initialization configuration
    """
    import torch
    import torch.nn as nn
    
    num_classes = len(registry)
    embeddings = nn.Embedding(num_classes, embedding_dim)
    
    if strategy == 'normal':
        nn.init.normal_(embeddings.weight, mean=0.0, std=0.02)
    elif strategy == 'orthogonal':
        nn.init.orthogonal_(embeddings.weight)
    elif strategy == 'zero':
        nn.init.zeros_(embeddings.weight)
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")
    
    return {
        'embedding_module': embeddings,
        'num_classes': num_classes,
        'embedding_dim': embedding_dim,
        'strategy': strategy
    }


def validate_label_consistency(
    registry: LabelRegistry,
    label_tensor: 'torch.Tensor'
) -> bool:
    """
    Validate that a label tensor contains only valid label IDs.
    
    Args:
        registry: Label registry to validate against
        label_tensor: Ground truth label tensor
        
    Returns:
        True if all labels are valid, False otherwise
    """
    import torch
    
    valid_ids = set(registry._id_to_name.keys())
    unique_labels = torch.unique(label_tensor).cpu().numpy().tolist()
    
    for label_id in unique_labels:
        if int(label_id) not in valid_ids:
            return False
    
    return True

