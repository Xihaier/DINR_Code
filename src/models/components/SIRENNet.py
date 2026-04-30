"""
Implementation of SIREN (Sinusoidal Representation Networks)

This module implements the paper:
'Implicit Neural Representations with Periodic Activation Functions'
(Sitzmann et al., 2020)

Reference:
    https://arxiv.org/abs/2006.09661
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import math


class SIRENLayer(nn.Module):
    """SIREN layer with sinusoidal activation function.
    
    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        omega_0 (float): Angular frequency factor
        is_first (bool): Whether this is the first layer
        linear (nn.Linear): Linear transformation layer
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ) -> None:
        """Initialize SIREN layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            omega_0: Angular frequency factor
            is_first: Whether this is the first layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights using uniform distribution."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.input_dim,
                    1 / self.input_dim
                )
            else:
                limit = math.sqrt(6 / self.input_dim) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sine activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after sinusoidal activation
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN network with stacked sinusoidal layers.
    
    Attributes:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        num_layers (int): Number of layers
        net (nn.Sequential): Sequential container of SIREN layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0
    ) -> None:
        """Initialize SIREN network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            
        Raises:
            ValueError: If num_layers < 2
        """
        super().__init__()
        
        if num_layers < 2:
            raise ValueError(
                f"Number of layers must be >= 2, got {num_layers}"
            )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Build network
        layers: List[nn.Module] = [
            SIRENLayer(
                input_dim,
                hidden_dim,
                omega_0=omega_0,
                is_first=True
            )
        ]
        
        for _ in range(num_layers - 2):
            layers.append(
                SIRENLayer(
                    hidden_dim,
                    hidden_dim,
                    omega_0=omega_0_hidden
                )
            )
        
        # Final linear layer
        final_linear = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            limit = math.sqrt(6 / hidden_dim) / omega_0_hidden
            final_linear.weight.uniform_(-limit, limit)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.net(x)
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def _test() -> None:
    """Test SIREN network implementation."""
    # Network parameters
    params = {
        "input_dim": 2,
        "hidden_dim": 256,
        "output_dim": 1,
        "num_layers": 7,
        "omega_0": 30.0,
        "omega_0_hidden": 30.0
    }
    
    # Create model and test input
    model = SIREN(**params)
    batch_size = 16
    example_input = torch.rand(batch_size, params["input_dim"])
    output = model(example_input)
    
    # Print model information
    trainable_params, total_params = model.get_param_count()
    
    print(f"Model Architecture:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nParameters:")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nShapes:")
    print(f"  Input: {example_input.shape}")
    print(f"  Output: {output.shape}")


if __name__ == "__main__":
    _test()