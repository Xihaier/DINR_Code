"""
Implementation of Dynamical SIREN
"""

from typing import Optional, Tuple, Literal
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


class SIRENBlock(nn.Module):
    """Basic SIREN block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        omega_0: float,
        dropout_rate: float = 0.0
    ) -> None:
        """Initialize SIREN block.
        
        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            dropout_rate: Dropout rate (applied after activation)
        """
        super().__init__()
        
        self.siren_layer = SIRENLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN block.
        
        Args:
            x: Input features
            
        Returns:
            Output features
        """
        out = self.siren_layer(x)
        return self.dropout(out)


class SIRENResidualBlock(nn.Module):
    """SIREN residual block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        omega_0: float,
        dropout_rate: float = 0.0
    ) -> None:
        """Initialize SIREN residual block.
        
        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.siren1 = SIRENLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False
        )
        
        self.siren2 = SIRENLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN residual block.
        
        Args:
            x: Input features
            
        Returns:
            Output features with residual connection
        """
        identity = x
        
        out = self.siren1(x)
        out = self.dropout(out)
        
        out = self.siren2(out)
        out = self.dropout(out)
        
        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t) using SIREN blocks with concatenation-based time conditioning."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        omega_0_hidden: float,
        dropout_rate: float,
        block_type: Literal["mlp", "residual"] = "residual"
    ) -> None:
        """Initialize SIREN-based ODE function.
        
        Args:
            dim: Feature dimension
            num_layers: Number of SIREN layers
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
        """
        super().__init__()
        
        # Input dimension includes time (dim + 1)
        block_dim = dim + 1
        
        # Choose block type
        Block = SIRENResidualBlock if block_type == "residual" else SIRENBlock

        # Build SIREN layers
        self.layers = nn.ModuleList([
            Block(
                dim=block_dim,
                omega_0=omega_0_hidden,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # Output projection to remove time dimension
        self.output_proj = nn.Linear(block_dim, dim)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through SIREN ODE function.
        
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
        
        # Pass through SIREN layers
        for layer in self.layers:
            x = layer(x)
            
        # Project back to original dimension
        return self.output_proj(x)


class DynamicalSIREN(nn.Module):
    """Dynamical SIREN."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        dropout_rate: float = 0.0,
        block_type: Literal["mlp", "residual"] = "residual",
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        final_activation: Optional[str] = None
    ) -> None:
        """Initialize the Dynamical SIREN model.
        
        Args:
            input_dim: Number of input dimensions
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_layers: Number of hidden layers in ODE function
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            final_activation: Optional activation for the output layer
            
        Raises:
            ValueError: If an unsupported activation name is provided
        """
        super().__init__()
        
        # Validate final activation
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
        
        if final_activation and final_activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(VALID_ACTIVATIONS.keys())}"
            )
            
        self.total_time = total_time
        self.num_steps = num_steps
        self.ot_lambda = ot_lambda
        
        # Initial embedding: z(0) = SIREN_first_layer(x)
        # Use SIREN's first layer initialization for input embedding
        self.input_embedding = SIRENLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            omega_0=omega_0,
            is_first=True  # Use first-layer initialization
        )

        # ODE function with SIREN dynamics and concatenation-only time conditioning
        self.ode_func = ODEFunc(
            dim=hidden_dim,
            num_layers=num_layers,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=dropout_rate,
            block_type=block_type
        )
                 
        # Output projection matching SIREN's final layer initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            limit = math.sqrt(6 / hidden_dim) / omega_0_hidden
            self.output_proj.weight.uniform_(-limit, limit)
            
        # Apply final activation if specified
        if final_activation:
            self.final_activation = VALID_ACTIVATIONS[final_activation]
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Dynamical SIREN.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (network_output, ot_regularization_term)
            - network_output: shape (batch_size, output_dim)  
            - ot_regularization_term: scalar tensor
        """
        # Set initial state z(0) = SIREN_embedding(x)
        z = self.input_embedding(x)
        
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
        output = self.output_proj(z)
        output = self.final_activation(output)
        
        return output, ot_reg
    
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
    """Run tests for the Dynamical SIREN network."""
    print("Testing Dynamical SIREN with concatenation-only time conditioning...")
    
    # Network parameters
    input_dim = 2
    hidden_dim = 192
    output_dim = 1
    num_layers = 7
    omega_0 = 30.0
    omega_0_hidden = 30.0
    dropout_rate = 0.0
    num_steps = 12
    final_activation = None
    
    # Test configurations
    configs = {
        "MLP Blocks": {"block_type": "mlp"},
        "Residual Blocks": {"block_type": "residual"}
    }

    for name, config in configs.items():
        print(f"\n--- Testing: {name} ---")
        
        model = DynamicalSIREN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            omega_0=omega_0,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=dropout_rate,
            block_type=config["block_type"],
            num_steps=num_steps,
            final_activation=final_activation
        )

        batch_size = 16
        x = torch.rand(batch_size, input_dim)
        y, ot_reg = model(x)
        
        trainable_params, total_params = model.get_param_count()
        
        print(f"Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num steps (M): {num_steps}")
        print(f"  Omega_0: {omega_0}")
        print(f"  Omega_0_hidden: {omega_0_hidden}")
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
