"""
Implementation of FINER (Flexible spectral-bias tuning in Implicit NEural Representations)

This module implements the paper:
'FINER: Flexible spectral-bias tuning in Implicit NEural Representation
by Variable-periodic Activation Functions'
(Liu et al., 2023)

Reference:
    https://arxiv.org/abs/2312.02434
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import math


class FINERLayer(nn.Module):
    """FINER layer with variable-periodic activation function.
    
    Uses a variable-periodic sinusoidal activation where the frequency
    is modulated by the input magnitude: sin(omega * scale * x) where
    scale = |Wx + b| + 1. This allows adaptive frequency tuning based
    on input values.
    
    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        omega_0 (float): Angular frequency factor
        is_first (bool): Whether this is the first layer
        scale_req_grad (bool): Whether scale computation requires gradient
        linear (nn.Linear): Linear transformation layer
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        first_bias_scale: Optional[float] = None,
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            omega_0: Angular frequency factor
            is_first: Whether this is the first layer
            first_bias_scale: Optional scale for first layer bias initialization
            scale_req_grad: Whether scale computation requires gradient
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first = is_first
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad = scale_req_grad
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.init_weights()
        
        if self.first_bias_scale is not None and self.is_first:
            self.init_first_bias()

    def init_weights(self) -> None:
        """Initialize weights using uniform distribution."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/input_dim, 1/input_dim]
                self.linear.weight.uniform_(
                    -1 / self.input_dim,
                    1 / self.input_dim
                )
            else:
                # Hidden layers: SIREN-style initialization
                limit = math.sqrt(6 / self.input_dim) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def init_first_bias(self) -> None:
        """Initialize first layer bias with specified scale."""
        with torch.no_grad():
            self.linear.bias.uniform_(
                -self.first_bias_scale,
                self.first_bias_scale
            )

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Generate adaptive scale factor based on input magnitude.
        
        Args:
            x: Linear transformation output
            
        Returns:
            Scale factor = |x| + 1
        """
        if self.scale_req_grad:
            return torch.abs(x) + 1
        else:
            with torch.no_grad():
                return torch.abs(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with variable-periodic sine activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after variable-periodic activation
        """
        linear_out = self.linear(x)
        scale = self.generate_scale(linear_out)
        return torch.sin(self.omega_0 * scale * linear_out)


class FINER(nn.Module):
    """FINER network with variable-periodic sinusoidal layers.
    
    Uses variable-periodic activation functions that adaptively tune
    the frequency based on input magnitude, allowing better control
    over spectral bias in implicit neural representations.
    
    Attributes:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        num_layers (int): Number of layers
        net (nn.Sequential): Sequential container of FINER layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        first_bias_scale: Optional[float] = None,
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            first_bias_scale: Optional scale for first layer bias
            scale_req_grad: Whether scale computation requires gradient
            
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
        layers = [
            FINERLayer(
                input_dim=input_dim,
                output_dim=hidden_dim,
                omega_0=omega_0,
                is_first=True,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad
            )
        ]
        
        for _ in range(num_layers - 2):
            layers.append(
                FINERLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    omega_0=omega_0_hidden,
                    is_first=False,
                    scale_req_grad=scale_req_grad
                )
            )
        
        # Final linear layer without activation
        final_linear = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            limit = math.sqrt(6 / hidden_dim) / omega_0_hidden
            final_linear.weight.uniform_(-limit, limit)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FINER network.
        
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
    """Test FINER network implementation."""
    # Network parameters
    params = {
        "input_dim": 2,
        "hidden_dim": 256,
        "output_dim": 1,
        "num_layers": 7,
        "omega_0": 30.0,
        "omega_0_hidden": 30.0,
        "first_bias_scale": None,
        "scale_req_grad": False
    }
    
    # Create model and test input
    model = FINER(**params)
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
    
    # Verify output shape
    assert output.shape == (batch_size, params["output_dim"]), \
        f"Expected {(batch_size, params['output_dim'])}, got {output.shape}"
    
    # Test with scale_req_grad=True
    print("\n--- Testing with scale_req_grad=True ---")
    params_grad = params.copy()
    params_grad["scale_req_grad"] = True
    model_grad = FINER(**params_grad)
    output_grad = model_grad(example_input)
    
    assert output_grad.shape == (batch_size, params["output_dim"]), \
        f"Expected {(batch_size, params['output_dim'])}, got {output_grad.shape}"
    
    print("✓ Test passed!")


if __name__ == "__main__":
    _test()
