"""
Implementation of Dynamical WIRE (Wavelet Implicit Neural Representations)

This module extends WIRE with ODE-based dynamics for improved spectral control.
"""

from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
import math


class ComplexGaborLayer(nn.Module):
    """Complex Gabor wavelet layer for WIRE networks.
    
    Uses a complex-valued Gabor wavelet activation function that combines
    a sinusoidal component with a Gaussian envelope.
    
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
                self.linear.weight.uniform_(
                    -1 / self.input_dim,
                    1 / self.input_dim
                )
            else:
                limit = math.sqrt(6 / self.input_dim) / self.omega_0.item()
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
        
        return torch.exp(1j * omega - scale.abs().square())


class WIREBlock(nn.Module):
    """Basic WIRE block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        omega_0: float,
        scale_0: float,
        dropout_rate: float = 0.0,
        trainable_params: bool = False
    ) -> None:
        """Initialize WIRE block.
        
        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            scale_0: Scale factor for Gaussian envelope
            dropout_rate: Dropout rate
            trainable_params: Whether omega and scale are trainable
        """
        super().__init__()
        
        self.wire_layer = ComplexGaborLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            scale_0=scale_0,
            is_first=False,
            trainable=trainable_params
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through WIRE block.
        
        Args:
            x: Input features (complex-valued)
            
        Returns:
            Output features (complex-valued)
        """
        out = self.wire_layer(x)
        # Apply dropout to real and imaginary parts
        if isinstance(self.dropout, nn.Dropout) and self.training:
            out_real = self.dropout(out.real)
            out_imag = self.dropout(out.imag)
            out = torch.complex(out_real, out_imag)
        return out


class WIREResidualBlock(nn.Module):
    """WIRE residual block for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        omega_0: float,
        scale_0: float,
        dropout_rate: float = 0.0,
        trainable_params: bool = False
    ) -> None:
        """Initialize WIRE residual block.
        
        Args:
            dim: Feature dimension
            omega_0: Angular frequency factor
            scale_0: Scale factor for Gaussian envelope
            dropout_rate: Dropout rate
            trainable_params: Whether omega and scale are trainable
        """
        super().__init__()
        
        self.wire1 = ComplexGaborLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            scale_0=scale_0,
            is_first=False,
            trainable=trainable_params
        )
        
        self.wire2 = ComplexGaborLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            scale_0=scale_0,
            is_first=False,
            trainable=trainable_params
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through WIRE residual block.
        
        Args:
            x: Input features (complex-valued)
            
        Returns:
            Output features with residual connection (complex-valued)
        """
        identity = x
        
        out = self.wire1(x)
        if isinstance(self.dropout, nn.Dropout) and self.training:
            out = torch.complex(self.dropout(out.real), self.dropout(out.imag))
        
        out = self.wire2(out)
        if isinstance(self.dropout, nn.Dropout) and self.training:
            out = torch.complex(self.dropout(out.real), self.dropout(out.imag))
        
        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t) using WIRE blocks with time conditioning.
    
    Operates in complex space with real-valued time concatenation.
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        omega_0_hidden: float,
        scale_0: float,
        dropout_rate: float,
        block_type: Literal["mlp", "residual"] = "residual",
        trainable_params: bool = False
    ) -> None:
        """Initialize WIRE-based ODE function.
        
        Args:
            dim: Feature dimension
            num_layers: Number of WIRE layers
            omega_0_hidden: Hidden layer frequency factor
            scale_0: Scale factor for Gaussian envelope
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            trainable_params: Whether omega and scale are trainable
        """
        super().__init__()
        
        # Input dimension includes time (dim + 1)
        block_dim = dim + 1
        
        # Choose block type
        Block = WIREResidualBlock if block_type == "residual" else WIREBlock
        
        # Build WIRE layers
        self.layers = nn.ModuleList([
            Block(
                dim=block_dim,
                omega_0=omega_0_hidden,
                scale_0=scale_0,
                dropout_rate=dropout_rate,
                trainable_params=trainable_params
            ) for _ in range(num_layers)
        ])
        
        # Output projection to remove time dimension (complex-valued)
        self.output_proj = nn.Linear(block_dim, dim, dtype=torch.cfloat)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through WIRE ODE function.
        
        Args:
            x: State tensor of shape (B, D), complex-valued
            t: Time scalar
            
        Returns:
            Time derivative dz/dt of shape (B, D), complex-valued
        """
        # Concatenate time to features (as complex with zero imaginary part)
        t_vec = torch.full(
            (x.shape[0], 1), t, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, t_vec], dim=1)
        
        # Pass through WIRE layers
        for layer in self.layers:
            x = layer(x)
            
        # Project back to original dimension
        return self.output_proj(x)


class DynamicalWIRE(nn.Module):
    """Dynamical WIRE network with ODE-based dynamics.
    
    Combines WIRE's complex Gabor wavelet representation with neural ODE
    dynamics for improved spectral control and optimal transport regularization.
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
        dropout_rate: float = 0.0,
        block_type: Literal["mlp", "residual"] = "residual",
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        trainable_params: bool = False,
        final_activation: Optional[str] = None
    ) -> None:
        """Initialize the Dynamical WIRE model.
        
        Args:
            input_dim: Number of input dimensions
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_layers: Number of hidden layers in ODE function
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            scale_0: Scale factor for Gaussian envelope
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            trainable_params: Whether omega and scale are trainable
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
        
        # Compensate for complex parameters
        self.hidden_dim = int(hidden_dim / math.sqrt(2))
        
        # Initial embedding: z(0) = WIRE_first_layer(x)
        self.input_embedding = ComplexGaborLayer(
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            omega_0=omega_0,
            scale_0=scale_0,
            is_first=True,
            trainable=trainable_params
        )

        # ODE function with WIRE dynamics and concatenation-only time conditioning
        self.ode_func = ODEFunc(
            dim=self.hidden_dim,
            num_layers=num_layers,
            omega_0_hidden=omega_0_hidden,
            scale_0=scale_0,
            dropout_rate=dropout_rate,
            block_type=block_type,
            trainable_params=trainable_params
        )
                 
        # Output projection (complex to real)
        self.output_proj = nn.Linear(self.hidden_dim, output_dim, dtype=torch.cfloat)
        with torch.no_grad():
            limit = math.sqrt(6 / self.hidden_dim) / omega_0_hidden
            self.output_proj.weight.real.uniform_(-limit, limit)
            self.output_proj.weight.imag.uniform_(-limit, limit)
            
        # Apply final activation if specified
        if final_activation:
            self.final_activation = VALID_ACTIVATIONS[final_activation]
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Dynamical WIRE.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (network_output, ot_regularization_term)
            - network_output: shape (batch_size, output_dim), real-valued
            - ot_regularization_term: scalar tensor
        """
        # Set initial state z(0) = WIRE_embedding(x)
        z = self.input_embedding(x)
        
        # Solve ODE using Euler's method
        ot_accum = 0.0
        dt = self.total_time / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            # Accumulate optimal transport regularization (use magnitude for complex)
            ot_accum = ot_accum + v.abs().pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum
        
        # Output projection from final state z(T)
        # Take real part of complex output
        output = self.output_proj(z).real
        output = self.final_activation(output)
        
        return output, ot_reg
    
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
            if p.is_complex():
                numel *= 2
            total += numel
            if p.requires_grad:
                trainable += numel
                
        return trainable, total


def _test():
    """Run tests for the Dynamical WIRE network."""
    print("Testing Dynamical WIRE with concatenation-only time conditioning...")
    
    # Network parameters
    input_dim = 2
    hidden_dim = 192
    output_dim = 1
    num_layers = 3
    omega_0 = 20.0
    omega_0_hidden = 20.0
    scale_0 = 10.0
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
        
        model = DynamicalWIRE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            omega_0=omega_0,
            omega_0_hidden=omega_0_hidden,
            scale_0=scale_0,
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
        print(f"  Hidden dim: {hidden_dim} (effective: {model.hidden_dim})")
        print(f"  Output dim: {output_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num steps (M): {num_steps}")
        print(f"  Omega_0: {omega_0}")
        print(f"  Omega_0_hidden: {omega_0_hidden}")
        print(f"  Scale_0: {scale_0}")
        print(f"  Block type: {config['block_type']}")
        print(f"\nParameters:")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"\nShapes:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        print(f"  Output dtype: {y.dtype}")
        print(f"  OT Reg value: {ot_reg.item():.4f}")
        
        # Verify output shape and dtype
        assert y.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {y.shape}"
        assert y.dtype == torch.float32, f"Expected float32 output, got {y.dtype}"
        assert ot_reg.numel() == 1, f"OT regularization should be scalar, got shape {ot_reg.shape}"
        
        print("✓ Test passed!")


if __name__ == "__main__":
    _test()
