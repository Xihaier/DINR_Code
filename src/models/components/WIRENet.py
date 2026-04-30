"""
Implementation of WIRE (Wavelet Implicit Neural Representations)

This module implements the paper:
'WIRE: Wavelet Implicit Neural Representations'
(Saragadam et al., 2023)

Reference:
    https://arxiv.org/abs/2301.05187
"""

from typing import Tuple
import torch
import torch.nn as nn
import math


class ComplexGaborLayer(nn.Module):
    """Complex Gabor wavelet layer for WIRE networks.
    
    Uses a complex-valued Gabor wavelet activation function that combines
    a sinusoidal component with a Gaussian envelope for better spectral
    control in implicit neural representations.
    
    Attributes:
        is_first (bool): Whether this is the first layer
        omega_0 (nn.Parameter): Angular frequency parameter
        scale_0 (nn.Parameter): Scale parameter for Gaussian envelope
        linear (nn.Linear): Linear transformation layer
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 10.0,
        scale_0: float = 10.0,
        is_first: bool = False,
        trainable: bool = False
    ) -> None:
        """Initialize Complex Gabor layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            omega_0: Angular frequency factor
            scale_0: Scale factor for Gaussian envelope
            is_first: Whether this is the first layer
            trainable: Whether omega and scale are trainable
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_first = is_first
        
        # Use real dtype for first layer, complex for subsequent layers
        dtype = torch.float if is_first else torch.cfloat
        
        # Gabor parameters
        self.omega_0 = nn.Parameter(
            omega_0 * torch.ones(1),
            requires_grad=trainable
        )
        self.scale_0 = nn.Parameter(
            scale_0 * torch.ones(1),
            requires_grad=trainable
        )
        
        self.linear = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights using uniform distribution."""
        with torch.no_grad():
            if self.is_first:
                # First layer uses standard initialization
                self.linear.weight.uniform_(
                    -1 / self.input_dim,
                    1 / self.input_dim
                )
            else:
                # Subsequent layers use SIREN-style initialization
                limit = math.sqrt(6 / self.input_dim) / self.omega_0.item()
                # Initialize real and imaginary parts separately for complex weights
                self.linear.weight.real.uniform_(-limit, limit)
                self.linear.weight.imag.uniform_(-limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with complex Gabor wavelet activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after Gabor wavelet activation
        """
        lin = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        # Complex Gabor: exp(i*omega - |scale|^2)
        return torch.exp(1j * omega - scale.abs().square())


class WIRE(nn.Module):
    """WIRE network with complex Gabor wavelet layers.
    
    Uses complex-valued intermediate representations with a Gabor wavelet
    activation function. The final output is the real part of the complex
    representation.
    
    Attributes:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension (before complex scaling)
        output_dim (int): Output dimension
        num_layers (int): Number of layers
        net (nn.Sequential): Sequential container of WIRE layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 20.0,
        omega_0_hidden: float = 20.0,
        scale_0: float = 10.0,
        trainable_params: bool = False
    ) -> None:
        """Initialize WIRE network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            scale_0: Scale factor for Gaussian envelope
            trainable_params: Whether omega and scale are trainable
            
        Raises:
            ValueError: If num_layers < 2
        """
        super().__init__()
        
        if num_layers < 2:
            raise ValueError(
                f"Number of layers must be >= 2, got {num_layers}"
            )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Compensate for complex parameters (each complex param = 2 real params)
        # Scale hidden_dim to keep parameter count comparable to real-valued networks
        self.hidden_dim = int(hidden_dim / math.sqrt(2))
        
        # Build network
        layers = [
            ComplexGaborLayer(
                input_dim=input_dim,
                output_dim=self.hidden_dim,
                omega_0=omega_0,
                scale_0=scale_0,
                is_first=True,
                trainable=trainable_params
            )
        ]
        
        for _ in range(num_layers - 2):
            layers.append(
                ComplexGaborLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    omega_0=omega_0_hidden,
                    scale_0=scale_0,
                    is_first=False,
                    trainable=trainable_params
                )
            )
        
        # Final complex-valued linear layer
        final_linear = nn.Linear(self.hidden_dim, output_dim, dtype=torch.cfloat)
        with torch.no_grad():
            limit = math.sqrt(6 / self.hidden_dim) / omega_0_hidden
            final_linear.weight.real.uniform_(-limit, limit)
            final_linear.weight.imag.uniform_(-limit, limit)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through WIRE network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim), real-valued
        """
        # Return real part of complex output
        return self.net(x).real
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Note: Complex parameters are counted as 2 real parameters each.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = 0
        total = 0
        
        for p in self.parameters():
            numel = p.numel()
            # Complex parameters have 2x the storage
            if p.is_complex():
                numel *= 2
            total += numel
            if p.requires_grad:
                trainable += numel
                
        return trainable, total


def _test() -> None:
    """Test WIRE network implementation."""
    # Network parameters
    params = {
        "input_dim": 2,
        "hidden_dim": 256,
        "output_dim": 1,
        "num_layers": 7,
        "omega_0": 20.0,
        "omega_0_hidden": 20.0,
        "scale_0": 10.0
    }
    
    # Create model and test input
    model = WIRE(**params)
    batch_size = 16
    example_input = torch.rand(batch_size, params["input_dim"])
    output = model(example_input)
    
    # Print model information
    trainable_params, total_params = model.get_param_count()
    
    print(f"Model Architecture:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"  Effective hidden_dim: {model.hidden_dim}")
    print(f"\nParameters:")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nShapes:")
    print(f"  Input: {example_input.shape}")
    print(f"  Output: {output.shape}")
    
    # Verify output is real
    assert output.dtype == torch.float32, f"Expected float32 output, got {output.dtype}"
    assert output.shape == (batch_size, params["output_dim"]), \
        f"Expected {(batch_size, params['output_dim'])}, got {output.shape}"
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    _test()
