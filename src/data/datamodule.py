import torch
import numpy as np
import imageio.v3 as iio
from typing import Any, Dict, Optional, List, Tuple, Set, Union, Callable
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn import functional as F
from omegaconf import DictConfig, ListConfig
import warnings
import os


class NormalizationError(ValueError):
    """Raised when data normalization fails."""
    pass


def normalize_data(data: torch.Tensor, method: str = "min-max") -> torch.Tensor:
    """Normalize data using the specified method.

    Args:
        data: Input tensor to normalize
        method: Normalization method ('min-max' or 'z-score')

    Returns:
        Normalized data tensor
        
    Raises:
        NormalizationError: If normalization fails
    """
    try:
        if method == "min-max":
            data_min, data_max = data.min(), data.max()
            if data_max - data_min < 1e-7:
                raise NormalizationError(
                    "Data has zero or near-zero range for min-max normalization"
                )
            return (data - data_min) / (data_max - data_min)

        elif method == "z-score":
            data_mean, data_std = data.mean(), data.std()
            if data_std < 1e-7:
                raise NormalizationError(
                    "Data has near-zero standard deviation for z-score normalization"
                )
            return (data - data_mean) / data_std
        elif method == None:
            return data
        else:
            raise ValueError(
                f"Invalid normalization method '{method}'. "
                "Supported options are 'min-max' and 'z-score'"
            )
            
    except Exception as e:
        if isinstance(e, (NormalizationError, ValueError)):
            raise
        raise NormalizationError(f"Normalization failed: {str(e)}")


class DataModule(LightningDataModule):
    """
    DataModule for Implicit Neural Representation.
    """
    def __init__(
        self,
        data_dir: str,
        in_features: int,
        normalization: str = "min-max",
        temporal: bool = False,
        data_shape: Optional[List[int]] = None,
        batch_size: Union[int, List[int]] = 8192,
        shuffle: Union[bool, List[bool]] = True,
        num_workers: Union[int, List[int]] = 4,
        pin_memory: Union[bool, List[bool]] = True,
        ntk_subset_mode: str = "subgrid",
        ntk_subgrid_g: int = 32,
        generalization_test: bool = False,
        generalization_train_percentage: float = 0.75,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.full_volume = None
        self.coord_vectors = None
        self.train_dataset, self.test_dataset = None, None

        self.data_dir = data_dir
        
        self._ntk_indices = None
        self._ntk_coords = None

        self.generalization_test = generalization_test
        self.generalization_train_percentage = generalization_train_percentage
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and process data."""
        if self.train_dataset is not None:
            return

        # Load and normalize the full data volume
        raw_data = torch.from_numpy(np.load(self.data_dir)).float()
        if self.hparams.data_shape:
            raw_data = raw_data.reshape(list(self.hparams.data_shape))
        
        self.full_volume = normalize_data(raw_data, self.hparams.normalization)

        # Create full-resolution coordinate vectors
        dims = self.full_volume.shape
        ranges = [(-1.0, 1.0)] * self.hparams.in_features
        if self.hparams.temporal:
            ranges[-1] = (0.0, 1.0)
        self.coord_vectors = [torch.linspace(r[0], r[1], d) for r, d in zip(ranges, dims)]

        coords_flat = torch.stack(torch.meshgrid(*self.coord_vectors, indexing="ij"), dim=-1).view(-1, self.hparams.in_features)
        targets_flat = self.full_volume.flatten().unsqueeze(-1)

        # Generalization test
        if self.generalization_test:
            num_total_samples = len(coords_flat)
            num_train_samples = int(num_total_samples * self.generalization_train_percentage)
            indices = torch.randperm(num_total_samples)
            train_indices = indices[:num_train_samples]
            test_indices = indices[num_train_samples:]
            self.train_dataset = TensorDataset(coords_flat[train_indices], targets_flat[train_indices])
            self.test_dataset = TensorDataset(coords_flat[test_indices], targets_flat[test_indices])
        else:
            full_pointwise_dataset = torch.utils.data.TensorDataset(coords_flat, targets_flat)
            self.train_dataset = full_pointwise_dataset
            self.test_dataset = full_pointwise_dataset

        # Build fixed NTK subset
        self._build_ntk_subset(coords_flat)

    def _create_dataloader(self, dataset: Dataset, dataloader_idx: int) -> DataLoader:
        """Helper function to create a DataLoader."""
        if dataloader_idx not in [0, 1]:
            raise IndexError(f"Invalid dataloader_idx: {dataloader_idx}. Must be 0 or 1.")
        
        # Determine if hyperparameters are specified per dataloader or globally
        is_list_batch_size = isinstance(self.hparams.batch_size, (list, ListConfig))
        is_list_shuffle = isinstance(self.hparams.shuffle, (list, ListConfig))
        is_list_num_workers = isinstance(self.hparams.num_workers, (list, ListConfig))
        is_list_pin_memory = isinstance(self.hparams.pin_memory, (list, ListConfig))

        # Get specific or global values for dataloader parameters
        batch_size = self.hparams.batch_size[dataloader_idx] if is_list_batch_size else self.hparams.batch_size
        shuffle = self.hparams.shuffle[dataloader_idx] if is_list_shuffle else (dataloader_idx == 0 and self.hparams.shuffle)
        num_workers = self.hparams.num_workers[dataloader_idx] if is_list_num_workers else self.hparams.num_workers
        pin_memory = self.hparams.pin_memory[dataloader_idx] if is_list_pin_memory else self.hparams.pin_memory

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )

    def _build_ntk_subset(self, coords_flat: torch.Tensor) -> None:
        """Build fixed NTK coordinate subset from the canonical grid.
        
        Args:
            coords_flat: Full flattened coordinate grid (N, in_features)
        """
        if self.hparams.ntk_subset_mode == "all":
            # Use all coordinates
            self._ntk_indices = torch.arange(len(coords_flat))
            self._ntk_coords = coords_flat.clone()
            print(f"✓ NTK subset: using all {len(coords_flat)} coordinates")
            return
        
        if self.hparams.ntk_subset_mode != "subgrid":
            warnings.warn(
                f"Unknown ntk_subset_mode '{self.hparams.ntk_subset_mode}', "
                f"falling back to 'subgrid'"
            )
        
        # Build uniform subgrid
        dims = self.full_volume.shape
        n_dims = len(dims)
        g = self.hparams.ntk_subgrid_g
        
        # Create uniform subgrid indices for each dimension
        subgrid_indices_per_dim = []
        for d, dim_size in enumerate(dims):
            if g >= dim_size:
                # If requested grid size >= actual size, use all points
                subgrid_indices_per_dim.append(torch.arange(dim_size))
            else:
                # Uniform spacing
                step = (dim_size - 1) / (g - 1) if g > 1 else 0
                indices = torch.round(torch.arange(g) * step).long()
                # Clamp to valid range
                indices = torch.clamp(indices, 0, dim_size - 1)
                subgrid_indices_per_dim.append(indices)
        
        # Build multi-dimensional subgrid using meshgrid (row-major order)
        subgrid_mesh = torch.meshgrid(*subgrid_indices_per_dim, indexing="ij")
        subgrid_coords_nd = torch.stack([mesh.flatten() for mesh in subgrid_mesh], dim=-1)
        
        # Convert ND indices to flat indices (row-major)
        strides = torch.tensor([np.prod(dims[i+1:], dtype=int) for i in range(n_dims)])
        flat_indices = (subgrid_coords_nd * strides).sum(dim=-1)
        
        # Store indices and coordinates
        self._ntk_indices = flat_indices
        self._ntk_coords = coords_flat[flat_indices].clone()
        
        print(f"✓ NTK subset: {len(self._ntk_coords)} coords from "
              f"{' × '.join(str(len(idx)) for idx in subgrid_indices_per_dim)} "
              f"uniform subgrid")
    
    def get_ntk_coords(self) -> Optional[torch.Tensor]:
        """Get fixed NTK coordinate subset (CPU tensor).
        
        Returns:
            NTK coordinates tensor (M, in_features) on CPU, or None if not set up
        """
        if self._ntk_coords is None:
            warnings.warn("NTK coordinates not yet initialized. Call setup() first.")
            return None
        return self._ntk_coords.clone()
    
    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader.
        """
        return self._create_dataloader(self.train_dataset, 0)

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader.
        """
        return self._create_dataloader(self.test_dataset, 1)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training or testing."""
        # Clear memory if needed
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the DataModule."""
        return {
            'data_dir': self.hparams.data_dir,
            'normalization': self.hparams.normalization,
            'temporal': self.hparams.temporal,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the DataModule."""
        # Update hyperparameters if needed
        for key, value in state_dict.items():
            if hasattr(self.hparams, key):
                setattr(self.hparams, key, value)


class ImageDataModule(LightningDataModule):
    """
    DataModule for image-based Implicit Neural Representation (e.g., Kodak images).
    
    Loads PNG/JPEG images and creates coordinate-to-RGB mappings for INR training.
    """
    def __init__(
        self,
        data_dir: str,
        image_name: str,
        in_features: int = 2,
        zero_mean: bool = False,
        batch_size: Union[int, List[int]] = 8192,
        shuffle: Union[bool, List[bool]] = True,
        num_workers: Union[int, List[int]] = 4,
        pin_memory: Union[bool, List[bool]] = True,
        ntk_subset_mode: str = "subgrid",
        ntk_subgrid_g: int = 32,
        generalization_test: bool = False,
        generalization_train_percentage: float = 0.75,
    ):
        """Initialize ImageDataModule.
        
        Args:
            data_dir: Base directory containing images (e.g., "data/kodak/")
            image_name: Name of the image file (e.g., "7.png")
            in_features: Number of input coordinate dimensions (default: 2 for images)
            zero_mean: If True, normalize to [-1, 1] instead of [0, 1]
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle training data
            num_workers: Number of dataloader workers
            pin_memory: Whether to pin memory in dataloaders
            ntk_subset_mode: Mode for NTK coordinate subset ("subgrid" or "all")
            ntk_subgrid_g: Grid size for NTK subgrid sampling
            generalization_test: Whether to split data for generalization testing
            generalization_train_percentage: Fraction of data for training if generalization_test=True
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.full_volume = None
        self.coord_vectors = None
        self.train_dataset, self.test_dataset = None, None
        
        # Construct full image path
        self.image_path = os.path.join(data_dir, image_name)
        
        self._ntk_indices = None
        self._ntk_coords = None
        
        # Image metadata (set during setup)
        self.image_height = None
        self.image_width = None
        self.image_channels = None
        
        self.generalization_test = generalization_test
        self.generalization_train_percentage = generalization_train_percentage
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and process image data."""
        if self.train_dataset is not None:
            return
        
        # Load image using imageio
        img = iio.imread(self.image_path)
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Handle grayscale images (add channel dimension)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        
        # Store image dimensions
        H, W, C = img.shape
        self.image_height = H
        self.image_width = W
        self.image_channels = C
        
        # Apply zero-mean normalization if requested: [0,1] -> [-1,1]
        if self.hparams.zero_mean:
            img = (img - 0.5) / 0.5
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        
        # Store full volume for compatibility with existing code
        # Note: for images, full_volume has shape (H, W, C), not (H, W) like .npy
        self.full_volume = img_tensor
        
        # Create coordinate vectors in [-1, 1]
        self.coord_vectors = [
            torch.linspace(-1.0, 1.0, H),
            torch.linspace(-1.0, 1.0, W)
        ]
        
        # Create coordinate grid: (H, W, 2) -> (H*W, 2)
        coords_flat = torch.stack(
            torch.meshgrid(*self.coord_vectors, indexing="ij"),
            dim=-1
        ).view(-1, self.hparams.in_features)
        
        # Flatten image to (H*W, C)
        targets_flat = img_tensor.view(-1, C)
        
        # Generalization test split
        if self.generalization_test:
            num_total_samples = len(coords_flat)
            num_train_samples = int(num_total_samples * self.generalization_train_percentage)
            indices = torch.randperm(num_total_samples)
            train_indices = indices[:num_train_samples]
            test_indices = indices[num_train_samples:]
            self.train_dataset = TensorDataset(coords_flat[train_indices], targets_flat[train_indices])
            self.test_dataset = TensorDataset(coords_flat[test_indices], targets_flat[test_indices])
        else:
            full_pointwise_dataset = TensorDataset(coords_flat, targets_flat)
            self.train_dataset = full_pointwise_dataset
            self.test_dataset = full_pointwise_dataset
        
        # Build fixed NTK subset
        self._build_ntk_subset(coords_flat)
        
        print(f"✓ Loaded image: {self.hparams.image_name} ({H}x{W}x{C})")
        print(f"  Normalization: /255.0" + (" + zero_mean" if self.hparams.zero_mean else ""))
        print(f"  Total samples: {len(coords_flat)}")
    
    def _create_dataloader(self, dataset: Dataset, dataloader_idx: int) -> DataLoader:
        """Helper function to create a DataLoader."""
        if dataloader_idx not in [0, 1]:
            raise IndexError(f"Invalid dataloader_idx: {dataloader_idx}. Must be 0 or 1.")
        
        # Determine if hyperparameters are specified per dataloader or globally
        is_list_batch_size = isinstance(self.hparams.batch_size, (list, ListConfig))
        is_list_shuffle = isinstance(self.hparams.shuffle, (list, ListConfig))
        is_list_num_workers = isinstance(self.hparams.num_workers, (list, ListConfig))
        is_list_pin_memory = isinstance(self.hparams.pin_memory, (list, ListConfig))

        # Get specific or global values for dataloader parameters
        batch_size = self.hparams.batch_size[dataloader_idx] if is_list_batch_size else self.hparams.batch_size
        shuffle = self.hparams.shuffle[dataloader_idx] if is_list_shuffle else (dataloader_idx == 0 and self.hparams.shuffle)
        num_workers = self.hparams.num_workers[dataloader_idx] if is_list_num_workers else self.hparams.num_workers
        pin_memory = self.hparams.pin_memory[dataloader_idx] if is_list_pin_memory else self.hparams.pin_memory

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def _build_ntk_subset(self, coords_flat: torch.Tensor) -> None:
        """Build fixed NTK coordinate subset from the canonical grid.
        
        Args:
            coords_flat: Full flattened coordinate grid (N, in_features)
        """
        if self.hparams.ntk_subset_mode == "all":
            self._ntk_indices = torch.arange(len(coords_flat))
            self._ntk_coords = coords_flat.clone()
            print(f"✓ NTK subset: using all {len(coords_flat)} coordinates")
            return
        
        if self.hparams.ntk_subset_mode != "subgrid":
            warnings.warn(
                f"Unknown ntk_subset_mode '{self.hparams.ntk_subset_mode}', "
                f"falling back to 'subgrid'"
            )
        
        # Build uniform subgrid for 2D image
        dims = (self.image_height, self.image_width)
        n_dims = len(dims)
        g = self.hparams.ntk_subgrid_g
        
        # Create uniform subgrid indices for each dimension
        subgrid_indices_per_dim = []
        for d, dim_size in enumerate(dims):
            if g >= dim_size:
                subgrid_indices_per_dim.append(torch.arange(dim_size))
            else:
                step = (dim_size - 1) / (g - 1) if g > 1 else 0
                indices = torch.round(torch.arange(g) * step).long()
                indices = torch.clamp(indices, 0, dim_size - 1)
                subgrid_indices_per_dim.append(indices)
        
        # Build multi-dimensional subgrid using meshgrid
        subgrid_mesh = torch.meshgrid(*subgrid_indices_per_dim, indexing="ij")
        subgrid_coords_nd = torch.stack([mesh.flatten() for mesh in subgrid_mesh], dim=-1)
        
        # Convert ND indices to flat indices (row-major)
        strides = torch.tensor([np.prod(dims[i+1:], dtype=int) for i in range(n_dims)])
        flat_indices = (subgrid_coords_nd * strides).sum(dim=-1)
        
        self._ntk_indices = flat_indices
        self._ntk_coords = coords_flat[flat_indices].clone()
        
        print(f"✓ NTK subset: {len(self._ntk_coords)} coords from "
              f"{' × '.join(str(len(idx)) for idx in subgrid_indices_per_dim)} "
              f"uniform subgrid")
    
    def get_ntk_coords(self) -> Optional[torch.Tensor]:
        """Get fixed NTK coordinate subset (CPU tensor).
        
        Returns:
            NTK coordinates tensor (M, in_features) on CPU, or None if not set up
        """
        if self._ntk_coords is None:
            warnings.warn("NTK coordinates not yet initialized. Call setup() first.")
            return None
        return self._ntk_coords.clone()
    
    def get_image_shape(self) -> Tuple[int, int, int]:
        """Get image dimensions (H, W, C).
        
        Returns:
            Tuple of (height, width, channels)
        """
        return (self.image_height, self.image_width, self.image_channels)
    
    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        return self._create_dataloader(self.train_dataset, 0)

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader."""
        return self._create_dataloader(self.test_dataset, 1)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training or testing."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the DataModule."""
        return {
            'image_path': self.image_path,
            'zero_mean': self.hparams.zero_mean,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'image_channels': self.image_channels,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the DataModule."""
        for key, value in state_dict.items():
            if hasattr(self.hparams, key):
                setattr(self.hparams, key, value)