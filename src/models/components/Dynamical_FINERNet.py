"""
Implementation of Dynamical FINER (Flexible spectral-bias tuning in Implicit NEural Representations)

This module extends FINER with ODE-based dynamics for improved spectral control.
"""

from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
import math


class FINERLayer(nn.Module):
    """FINER layer with variable-periodic activation function.
    
    Uses a variable-periodic sinusoidal activation where the frequency
    is modulated by the input magnitude: sin(omega * scale * x) where
    scale = |Wx + b| + 1.
    
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
                self.linear.weight.uniform_(
                    -1 / self.input_dim,
                    1 / self.input_dim
                )
            else:
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


class FINERBlock(nn.Module):
    """Basic FINER block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        omega_0: float,
        dropout_rate: float = 0.0,
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER block.
        
        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            dropout_rate: Dropout rate (applied after activation)
            scale_req_grad: Whether scale computation requires gradient
        """
        super().__init__()
        
        self.finer_layer = FINERLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False,
            scale_req_grad=scale_req_grad
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FINER block.
        
        Args:
            x: Input features
            
        Returns:
            Output features
        """
        out = self.finer_layer(x)
        return self.dropout(out)


class FINERResidualBlock(nn.Module):
    """FINER residual block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        omega_0: float,
        dropout_rate: float = 0.0,
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER residual block.
        
        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            dropout_rate: Dropout rate
            scale_req_grad: Whether scale computation requires gradient
        """
        super().__init__()
        
        self.finer1 = FINERLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False,
            scale_req_grad=scale_req_grad
        )
        
        self.finer2 = FINERLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False,
            scale_req_grad=scale_req_grad
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FINER residual block.
        
        Args:
            x: Input features
            
        Returns:
            Output features with residual connection
        """
        identity = x
        
        out = self.finer1(x)
        out = self.dropout(out)
        
        out = self.finer2(out)
        out = self.dropout(out)
        
        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t) using FINER blocks with concatenation-based time conditioning."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        omega_0_hidden: float,
        dropout_rate: float,
        block_type: Literal["mlp", "residual"] = "residual",
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER-based ODE function.
        
        Args:
            dim: Feature dimension
            num_layers: Number of FINER layers
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            scale_req_grad: Whether scale computation requires gradient
        """
        super().__init__()
        
        # Input dimension includes time (dim + 1)
        block_dim = dim + 1
        
        # Choose block type
        Block = FINERResidualBlock if block_type == "residual" else FINERBlock

        # Build FINER layers
        self.layers = nn.ModuleList([
            Block(
                dim=block_dim,
                omega_0=omega_0_hidden,
                dropout_rate=dropout_rate,
                scale_req_grad=scale_req_grad
            ) for _ in range(num_layers)
        ])
        
        # Output projection to remove time dimension
        self.output_proj = nn.Linear(block_dim, dim)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through FINER ODE function.
        
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
        
        # Pass through FINER layers
        for layer in self.layers:
            x = layer(x)
            
        # Project back to original dimension
        return self.output_proj(x)


class DynamicalFINER(nn.Module):
    """Dynamical FINER network with ODE-based dynamics.
    
    Combines FINER's variable-periodic activation with neural ODE
    dynamics for improved spectral control and optimal transport regularization.
    """
    
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
        first_bias_scale: Optional[float] = None,
        scale_req_grad: bool = False,
        final_activation: Optional[str] = None
    ) -> None:
        """Initialize the Dynamical FINER model.
        
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
            first_bias_scale: Optional scale for first layer bias
            scale_req_grad: Whether scale computation requires gradient
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
        self.hidden_dim = hidden_dim
        
        # Initial embedding: z(0) = FINER_first_layer(x)
        self.input_embedding = FINERLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            omega_0=omega_0,
            is_first=True,
            first_bias_scale=first_bias_scale,
            scale_req_grad=scale_req_grad
        )

        # ODE function with FINER dynamics and concatenation-only time conditioning
        self.ode_func = ODEFunc(
            dim=hidden_dim,
            num_layers=num_layers,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=dropout_rate,
            block_type=block_type,
            scale_req_grad=scale_req_grad
        )
                 
        # Output projection matching FINER's final layer initialization
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
        """Forward pass of the Dynamical FINER.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (network_output, ot_regularization_term)
            - network_output: shape (batch_size, output_dim)  
            - ot_regularization_term: scalar tensor
        """
        # Set initial state z(0) = FINER_embedding(x)
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
    """Run tests for the Dynamical FINER network."""
    print("Testing Dynamical FINER with concatenation-only time conditioning...")
    
    # Network parameters
    input_dim = 2
    hidden_dim = 192
    output_dim = 1
    num_layers = 3
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
        
        model = DynamicalFINER(
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
    
    # Test with scale_req_grad=True
    print("\n--- Testing with scale_req_grad=True ---")
    model_grad = DynamicalFINER(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        omega_0=omega_0,
        omega_0_hidden=omega_0_hidden,
        dropout_rate=dropout_rate,
        block_type="residual",
        num_steps=num_steps,
        scale_req_grad=True,
        final_activation=final_activation
    )
    
    y_grad, ot_reg_grad = model_grad(x)
    assert y_grad.shape == (batch_size, output_dim)
    print("✓ Test passed!")


if __name__ == "__main__":
    _test()
