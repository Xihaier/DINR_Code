"""
Implementation of Dynamical FFNet
"""

from typing import Optional, Tuple, Literal
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
    """Basic MLP block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        dropout_rate: float,
        activation: nn.Module
    ) -> None:
        """Initialize MLP block.
        
        Args:
            dim: Feature dimension
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP block.
        
        Args:
            x: Input features
            
        Returns:
            Output features
        """
        out = self.norm(x)
        out = self.linear(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        dropout_rate: float,
        activation: nn.Module
    ) -> None:
        """Initialize residual block.
        
        Args:
            dim: Feature dimension
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input features
            
        Returns:
            Output features with residual connection
        """
        identity = x
        
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.dropout(out)
        
        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t) with concatenation-based time conditioning."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: nn.Module,
        block_type: Literal["mlp", "residual"] = "residual"
    ) -> None:
        """Initialize ODE function.
        
        Args:
            dim: Feature dimension
            num_layers: Number of layers
            dropout_rate: Dropout rate
            activation: Activation function
            block_type: Type of block ("mlp" or "residual")
        """
        super().__init__()
        
        # Input dimension includes time (dim + 1)
        block_dim = dim + 1
        
        # Choose block type
        Block = MLPBlock if block_type == "mlp" else ResidualBlock
        
        # Build layers
        self.layers = nn.ModuleList([
            Block(
                dim=block_dim,
                dropout_rate=dropout_rate,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # Output projection to remove time dimension
        self.output_proj = nn.Linear(block_dim, dim)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through ODE function.
        
        Args:
            x: State tensor of shape (B, D)
            t: Time scalar
            
        Returns:
            Time derivative dz/dt of shape (B, D)
        """
        # Concatenate time to features
        t_vec = torch.full(
            (x.shape[0], 1), t, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, t_vec], dim=1)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
            
        # Project back to original dimension
        return self.output_proj(x)


class DynamicalFourierFeatureNetwork(nn.Module):
    """Dynamical Fourier Feature Network."""
    
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
        output_dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: str,
        block_type: Literal["mlp", "residual"] = "residual",
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        sigma: float = 1.0,
        final_activation: Optional[str] = None
    ) -> None:
        """Initialize the Dynamical Fourier Feature Network model.
        
        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of Fourier feature mapping
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_layers: Number of layers in ODE function
            dropout_rate: Dropout rate
            activation: Activation function name
            block_type: Type of block ("mlp" or "residual")
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            sigma: Standard deviation for Fourier features
            final_activation: Optional activation for the output layer
            
        Raises:
            ValueError: If an unsupported activation name is provided
        """
        super().__init__()
        
        if final_activation and final_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(self.VALID_ACTIVATIONS.keys())}"
            )
            
        self.total_time = total_time
        self.num_steps = num_steps
        self.ot_lambda = ot_lambda
        
        # Initial embedding: z(0) = phi(x)
        self.fourier_features = torch.jit.script(FourierFeatureMapping(
            input_dim=input_dim,
            mapping_size=mapping_size,
            sigma=sigma
        ))

        # Projection from mapping_size to hidden_dim if needed
        if mapping_size != hidden_dim:
            self.input_proj = nn.Linear(mapping_size, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # ODE function with concatenation-only time conditioning
        self.ode_func = ODEFunc(
            dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            activation=self._get_activation(activation),
            block_type=block_type
        )
                 
        # Output projection
        output_layers = [nn.Linear(hidden_dim, output_dim)]
        if final_activation:
            output_layers.append(self._get_activation(final_activation))
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Dynamical Fourier Feature Network.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (network_output)
            - network_output: shape (batch_size, output_dim)
        """
        # Set initial state z(0) = phi(x)
        z = self.fourier_features(x)
        z = self.input_proj(z)
        
        # Solve ODE using Euler's method
        ot_accum = 0.0
        dt = self.total_time / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            # Accumulate optimal transport regularization
            ot_accum = ot_accum + v.pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum
        
        # Output projection from final state z(T)
        return self.output_proj(z), ot_reg
    
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
    """Run tests for the Dynamical Fourier Feature Network."""
    print("Testing Dynamical Fourier Feature Network")
    
    # Network parameters
    input_dim = 2
    mapping_size = 256
    hidden_dim = 256
    output_dim = 1
    num_layers = 3
    dropout_rate = 0.1
    activation = "GELU"
    num_steps = 10
    sigma = 10.0
    final_activation = None
    
    # Test configurations
    configs = {
        "MLP Blocks": {"block_type": "mlp"},
        "Residual Blocks": {"block_type": "residual"}
    }

    for name, config in configs.items():
        print(f"\n--- Testing: {name} ---")
        
        model = DynamicalFourierFeatureNetwork(
            input_dim=input_dim,
            mapping_size=mapping_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            block_type=config["block_type"],
            num_steps=num_steps,
            sigma=sigma,
            final_activation=final_activation
        )

        batch_size = 16
        x = torch.rand(batch_size, input_dim)
        y, ot_reg = model(x)
        
        trainable_params, total_params = model.get_param_count()
        
        print(f"Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Mapping size: {mapping_size}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num steps (M): {num_steps}")
        print(f"  Block type: {config['block_type']}")
        print(f"\nParameters:")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"\nShapes:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        print(f"  OT Reg value: {ot_reg.item():.4f}")
        
        # Verify output shape
        assert y.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {y.shape}"
        assert ot_reg.numel() == 1, f"OT regularization should be scalar, got shape {ot_reg.shape}"
        
        print("✓ Test passed!")


if __name__ == "__main__":
    _test()