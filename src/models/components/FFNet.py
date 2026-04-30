"""
Implementation of Fourier Feature Networks.

This module implements the paper:
'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains'
(Tancik et al., 2020)

Reference:
    https://arxiv.org/abs/2006.10739
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping module for input coordinate lifting.
    
    Maps input coordinates to a higher dimensional space using random Fourier features,
    enabling better learning of high-frequency functions.
    
    Attributes:
        input_dim (int): Dimensionality of input coordinates
        mapping_size (int): Output dimension of the Fourier mapping
        sigma (float): Standard deviation for feature sampling
        B (nn.Parameter): Random Fourier feature matrix
    """
    
    def __init__(
        self, 
        input_dim: int,
        mapping_size: int,
        sigma: float = 1.0
    ) -> None:
        """Initialize Fourier feature mapping.
        
        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of the feature mapping (must be even)
            sigma: Standard deviation for sampling feature matrix
            
        Raises:
            ValueError: If mapping_size is not even
        """
        super().__init__()
        
        if mapping_size % 2 != 0:
            raise ValueError(
                f"mapping_size must be even, got {mapping_size}"
            )
            
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.sigma = sigma
        
        # Initialize random Fourier features
        self.B = nn.Parameter(
            torch.randn(input_dim, mapping_size // 2) * sigma,
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping to input coordinates.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Fourier features of shape (batch_size, mapping_size)
            
        Raises:
            ValueError: If input dimensions don't match expected shape
        """
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(-1)}"
            )
            
        # Project and apply sinusoidal activation
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPBlock(nn.Module):
    """Basic MLP block with optional layer normalization and dropout."""
    
    def __init__(
        self,
        dim: int,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = FourierFeatureNetwork._get_activation(activation)
        self.layer_norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(x)
        out = self.linear(out)
        out = self.activation(out)
        return self.dropout(out)


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization design."""
    
    def __init__(
        self,
        dim: int,
        dropout_rate: float = 0.1,
        activation: str = "GELU"
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = FourierFeatureNetwork._get_activation(activation)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layer_norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return identity + out


class FourierFeatureNetwork(nn.Module):
    """Neural network with Fourier features and optional residual connections."""
    
    VALID_ACTIVATIONS = {
        "ReLU": nn.ReLU(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "ELU": nn.ELU(),
        "SELU": nn.SELU(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity()
    }
    
    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        sigma: float = 1.0,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        final_activation: Optional[str] = None,
        use_residual: bool = True
    ) -> None:
        """Initialize network with optional residual connections.
        
        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of Fourier feature mapping
            hidden_dim: Width of hidden layers
            num_layers: Number of hidden layers
            output_dim: Number of output dimensions
            sigma: Standard deviation for Fourier features
            dropout_rate: Dropout probability
            activation: Activation function for hidden layers
            final_activation: Optional activation function for output layer
            use_residual: Whether to use residual connections
            
        Raises:
            ValueError: If activation name is not supported
        """
        super().__init__()
        
        if activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from {list(self.VALID_ACTIVATIONS.keys())}"
            )
        
        if final_activation and final_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(self.VALID_ACTIVATIONS.keys())}"
            )
            
        # Use torch.jit.script for Fourier feature computation
        self.fourier_features = torch.jit.script(FourierFeatureMapping(
            input_dim=input_dim,
            mapping_size=mapping_size,
            sigma=sigma
        ))
        
        # Use faster activation functions
        self.activation = nn.ReLU()  # or nn.GELU()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(mapping_size, hidden_dim),
            self.activation,
            nn.Dropout(dropout_rate)
        )
        
        # Hidden layers with specified activation
        if use_residual:
            self.hidden_blocks = nn.ModuleList([
                ResidualBlock(
                    dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    activation=activation
                ) for _ in range(num_layers)
            ])
        else:
            self.hidden_blocks = nn.ModuleList([
                MLPBlock(
                    dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    use_layer_norm=True
                ) for _ in range(num_layers)
            ])
        
        # Output projection with optional final activation
        output_layers = [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ]
        if final_activation:
            output_layers.append(self.VALID_ACTIVATIONS[final_activation])
        self.output_proj = nn.Sequential(*output_layers)
    
    @classmethod
    def _get_activation(cls, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        if activation_name not in cls.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation_name}. "
                f"Choose from {list(cls.VALID_ACTIVATIONS.keys())}"
            )
        return cls.VALID_ACTIVATIONS[activation_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Network output of shape (batch_size, output_dim)
        """
        # Fourier feature mapping
        x = self.fourier_features(x)
        
        # Input projection
        x = self.input_proj(x)
        
        # Hidden layers
        for block in self.hidden_blocks:
            x = block(x)
            
        # Output projection
        return self.output_proj(x)
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params


def _test():
    """Run tests for the Fourier feature network."""
    # Network parameters
    input_dim = 2
    mapping_size = 256
    hidden_dim = 256
    num_layers = 5
    output_dim = 1
    sigma = 10.0
    dropout_rate = 0.1
    activation = "GELU"
    final_activation = None
    
    # Test both residual and non-residual versions
    for use_residual in [True, False]:
        print(f"\nTesting {'Residual' if use_residual else 'MLP'} version:")
        
        model = FourierFeatureNetwork(
            input_dim=input_dim,
            mapping_size=mapping_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            sigma=sigma,
            dropout_rate = dropout_rate,
            activation = activation,
            final_activation = final_activation,
            use_residual=use_residual
        )

        batch_size = 16
        x = torch.rand(batch_size, input_dim)
        y = model(x)
        
        trainable_params, total_params = model.get_param_count()
        
        print(f"Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Mapping size: {mapping_size}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Output dim: {output_dim}")
        print(f"  Using residual: {use_residual}")
        print(f"\nParameters:")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"\nShapes:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")


if __name__ == "__main__":
    _test()