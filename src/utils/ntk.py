"""
Neural Tangent Kernel (NTK) Analysis Pipeline

References:
    - Neural Tangent Kernel: Convergence and Generalization in Neural Networks (Jacot et al., 2018)
    - On the Inductive Bias of Neural Tangent Kernels (Bietti & Mairal, 2019)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class NTKResult:
    """Container for NTK analysis results.
    
    Attributes:
        eigenvalues: All eigenvalues sorted in descending order
        top_k_eigenvalues: Top-k eigenvalues for convenient access (subset of eigenvalues)
        eigenvectors: Corresponding eigenvectors (optional)
        condition_number: λ_max / λ_min (for positive eigenvalues)
        trace: Sum of all eigenvalues
        effective_rank: Participation ratio (trace²/||λ||²)
        explained_variance: Cumulative explained variance ratio
        spectrum_decay: Rate of eigenvalue decay
        gram_matrix: Full NTK matrix (optional, for small datasets)
        metadata: Additional analysis metadata
    """
    eigenvalues: np.ndarray  # All eigenvalues
    top_k_eigenvalues: np.ndarray  # Top-k subset for convenience
    condition_number: float
    trace: float
    effective_rank: float
    explained_variance: np.ndarray
    spectrum_decay: float
    eigenvectors: Optional[np.ndarray] = None
    gram_matrix: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class NTKAnalyzer:
    """Neural Tangent Kernel analyzer for OCINR models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        default_top_k: int = 10
    ):
        """Initialize NTK analyzer.
        
        Args:
            model: Neural network model to analyze
            device: Device to run computations on (auto-detected if None)
            dtype: Computation dtype for numerical stability
            default_top_k: Default number of top eigenvalues to return
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.default_top_k = default_top_k
        
        # Move model to specified device and dtype
        self.model = self.model.to(self.device, dtype=self.dtype)
        
        # Cache trainable parameters
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        if len(self._params) == 0:
            raise ValueError("Model has no trainable parameters")
        
        self._param_shapes = [p.shape for p in self._params]
        self._param_sizes = [p.numel() for p in self._params]
        self._total_params = sum(self._param_sizes)
        
    def _extract_prediction(self, output: Union[Tensor, Tuple, List]) -> Tensor:
        """Extract prediction tensor from model output.
        
        Handles models that return:
        - Single tensor: y
        - Tuple/List: (y, aux_terms...)
        
        Args:
            output: Model output
            
        Returns:
            Prediction tensor of shape (N, C)
        """
        if isinstance(output, (tuple, list)):
            y = output[0]
        else:
            y = output
            
        # Ensure 2D shape (N, C)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim > 2:
            raise ValueError(f"Expected 1D or 2D output, got shape {y.shape}")
            
        return y
    
    def _flatten_gradients(
        self, 
        gradients: List[Optional[Tensor]]
    ) -> Tensor:
        """Flatten gradients into a single vector.
        
        Args:
            gradients: List of per-parameter gradients (may contain None)
            
        Returns:
            Flattened gradient vector
        """
        flat_grads = []
        for grad, size in zip(gradients, self._param_sizes):
            if grad is None:
                # Handle unused parameters with zero gradients
                flat_grads.append(torch.zeros(size, device=self.device, dtype=self.dtype))
            else:
                flat_grads.append(grad.reshape(-1))
                
        return torch.cat(flat_grads, dim=0)
    
    def compute_ntk_matrix(
        self,
        inputs: Tensor,
        *,
        normalize: Literal["none", "trace", "params", "params_outputs"] = "none",
        center_kernel: bool = False,
        chunk_size: Optional[int] = None
    ) -> Tensor:
        """Compute empirical NTK matrix.
        
        The empirical NTK is defined as:
        K_ij = Σ_c ⟨∂f_c(x_i)/∂θ, ∂f_c(x_j)/∂θ⟩
        
        Args:
            inputs: Input coordinates of shape (N, input_dim)
            normalize: Normalization scheme for the kernel
            center_kernel: Whether to apply kernel centering
            chunk_size: Process inputs in chunks to save memory (NOT IMPLEMENTED)
            
        Returns:
            NTK matrix of shape (N, N) on CPU
        """
        if chunk_size is not None:
            raise NotImplementedError(
                "Chunked NTK computation is not implemented. "
                "For large datasets, reduce ntk_subgrid_g or use ntk_subset_mode='subgrid' "
                "with a smaller grid size in your DataModule configuration."
            )
        
        inputs = inputs.to(self.device, dtype=self.dtype)
        N = inputs.shape[0]
        
        # Store original training state and set to eval
        was_training = self.model.training
        self.model.eval()
        
        try:
            return self._compute_ntk_standard(
                inputs, normalize, center_kernel
            )
        finally:
            # Restore training state
            self.model.train(was_training)
    
    def _compute_ntk_standard(
        self,
        inputs: Tensor,
        normalize: str,
        center_kernel: bool
    ) -> Tensor:
        """Standard NTK computation for moderate-sized datasets."""
        N = inputs.shape[0]
        
        # Forward pass with gradient tracking
        with torch.enable_grad():
            inputs.requires_grad_(True)
            output = self.model(inputs)
            y = self._extract_prediction(output)
            N, C = y.shape
            
            # Compute Jacobian matrix J ∈ R^{(N×C) × P}
            jacobian_rows = []
            total_outputs = N * C
            
            for i in range(N):
                for c in range(C):
                    # Compute gradients for output y[i, c]
                    retain_graph = not (i == N - 1 and c == C - 1)
                    grads = torch.autograd.grad(
                        outputs=y[i, c],
                        inputs=self._params,
                        retain_graph=retain_graph,
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    # Flatten and store gradient
                    grad_flat = self._flatten_gradients(grads)
                    jacobian_rows.append(grad_flat.detach().cpu())
            
            # Stack into Jacobian matrix
            J = torch.stack(jacobian_rows, dim=0)  # Shape: (N×C, P)
        
        # # Compute NTK: K = J @ J^T
        # K = J @ J.t()  # Shape: (N×C, N×C)
        
        # # Sum over output channels to get (N, N) kernel
        # if C > 1:
        #     K = K.view(N, C, N, C).sum(dim=(1, 3))  # Sum over output dimensions
        # else:
        #     K = K.squeeze()
            
        # Reshape Jacobian to (N, C, P)
        P = J.shape[1]
        J = J.view(N, C, P)  # (N, C, P)

        # Empirical NTK: K_ij = sum_c <J[i,c,:], J[j,c,:]>
        K = torch.einsum('icp,jcp->ij', J, J)  # (N, N)        

        # Apply normalization
        K = self._apply_normalization(K, normalize, C)
        
        # Apply kernel centering if requested
        if center_kernel:
            K = self._center_kernel(K)
            
        # Ensure numerical symmetry
        K = 0.5 * (K + K.t())
        
        return K.detach()
    
    
    def _apply_normalization(self, K: Tensor, normalize: str, num_outputs: int) -> Tensor:
        """Apply normalization to the kernel matrix."""
        if normalize == "none":
            return K
        elif normalize == "trace":
            trace = torch.trace(K)
            return K / (trace + 1e-12)
        elif normalize == "params":
            return K / self._total_params
        elif normalize == "params_outputs":
            return K / (self._total_params * num_outputs)
        else:
            raise ValueError(f"Unknown normalization: {normalize}")
    
    def _center_kernel(self, K: Tensor) -> Tensor:
        """Apply kernel centering: K_centered = H @ K @ H."""
        N = K.shape[0]
        H = torch.eye(N, dtype=K.dtype) - torch.ones(N, N, dtype=K.dtype) / N
        return H @ K @ H
    
    def analyze_spectrum(
        self,
        inputs: Tensor,
        *,
        top_k: Optional[int] = None,
        normalize: Literal["none", "trace", "params", "params_outputs"] = "none",
        center_kernel: bool = False,
        return_eigenvectors: bool = False,
        return_matrix: bool = False,
        return_all_eigenvalues: bool = True
    ) -> NTKResult:
        """Perform comprehensive NTK spectrum analysis.
        
        Args:
            inputs: Input coordinates
            top_k: Number of top eigenvalues to keep for top_k_eigenvalues field (None = all)
            normalize: Kernel normalization scheme
            center_kernel: Whether to center the kernel
            return_eigenvectors: Whether to compute and return eigenvectors
            return_matrix: Whether to return the full kernel matrix
            return_all_eigenvalues: Always True; kept for API compatibility (all eigenvalues are always returned)
            
        Returns:
            NTKResult with comprehensive spectrum analysis
        
        Note:
            return_all_eigenvalues is kept for backward compatibility but is always True.
            The full spectrum is always computed and stored in NTKResult.eigenvalues.
        """
        # Use default top_k if not specified
        if top_k is None:
            top_k = self.default_top_k
        
        # Compute kernel matrix (model put in eval mode inside)
        K = self.compute_ntk_matrix(
            inputs,
            normalize=normalize,
            center_kernel=center_kernel
        )
        
        N = K.shape[0]
        
        # Eigendecomposition (use double precision for stability)
        K_double = K.to(torch.float64)
        if return_eigenvectors:
            eigenvals, eigenvecs = torch.linalg.eigh(K_double)
            # Sort in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            eigenvecs = eigenvecs.numpy() if return_eigenvectors else None
        else:
            eigenvals = torch.linalg.eigvalsh(K_double)
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            eigenvecs = None
        
        eigenvals = eigenvals.numpy()
        
        # Sanity checks
        if len(eigenvals) != N:
            warnings.warn(f"Expected {N} eigenvalues but got {len(eigenvals)}")
        
        if normalize == "trace" and not center_kernel:
            trace_val = eigenvals.sum()
            if not (0.99 < trace_val < 1.01):
                warnings.warn(
                    f"Trace normalization sanity check failed: trace = {trace_val:.6f} (expected ~1.0)"
                )
        
        # Store all eigenvalues
        all_eigenvals = eigenvals.copy()
        
        # Get top_k eigenvalues for convenience (but keep all in eigenvalues field)
        if top_k is not None and top_k < len(eigenvals):
            top_k_eigenvals = eigenvals[:top_k]
            if eigenvecs is not None:
                top_k_eigenvecs = eigenvecs[:, :top_k]
            else:
                top_k_eigenvecs = None
        else:
            top_k_eigenvals = eigenvals.copy()
            top_k_eigenvecs = eigenvecs
        
        # Compute spectrum statistics on all eigenvalues
        stats = self._compute_spectrum_statistics(all_eigenvals)
        
        # Create result with ALL eigenvalues stored
        result = NTKResult(
            eigenvalues=all_eigenvals,  # Store ALL eigenvalues
            top_k_eigenvalues=top_k_eigenvals,  # Store top-k for convenience
            eigenvectors=top_k_eigenvecs,  # Only top-k eigenvectors to save memory
            condition_number=stats["condition_number"],
            trace=stats["trace"],
            effective_rank=stats["effective_rank"],
            explained_variance=stats["explained_variance"],
            spectrum_decay=stats["spectrum_decay"],
            gram_matrix=K.numpy() if return_matrix else None,
            metadata={
                "num_samples": inputs.shape[0],
                "num_params": self._total_params,
                "normalization": normalize,
                "centered": center_kernel,
                "input_dim": inputs.shape[1],
                "model_type": type(self.model).__name__,
                "top_k": top_k,
                "total_eigenvalues": len(all_eigenvals)
            }
        )
        
        return result
    
    def _compute_spectrum_statistics(self, eigenvals: np.ndarray) -> Dict:
        """Compute comprehensive spectrum statistics."""
        eps = 1e-12
        positive_eigs = eigenvals[eigenvals > eps]
        
        # Basic statistics
        trace = float(eigenvals.sum())
        max_eig = float(eigenvals[0]) if len(eigenvals) > 0 else 0.0
        min_pos_eig = float(positive_eigs[-1]) if len(positive_eigs) > 0 else eps
        
        # Condition number
        condition_number = max_eig / min_pos_eig if len(positive_eigs) > 0 else float('inf')
        
        # Effective rank (participation ratio)
        sum_eigs = eigenvals.sum()
        sum_eigs_sq = (eigenvals ** 2).sum()
        effective_rank = (sum_eigs ** 2) / (sum_eigs_sq + eps)
        
        # Explained variance ratio
        explained_variance = np.cumsum(eigenvals) / (sum_eigs + eps)
        
        # Spectrum decay rate (fit exponential decay)
        if len(eigenvals) > 2:
            # Fit log(λ_i) ≈ a - b*i
            indices = np.arange(1, len(eigenvals) + 1)
            log_eigs = np.log(np.maximum(eigenvals, eps))
            spectrum_decay = -np.polyfit(indices, log_eigs, 1)[0]
        else:
            spectrum_decay = 0.0
        
        return {
            "condition_number": condition_number,
            "trace": trace,
            "effective_rank": effective_rank,
            "explained_variance": explained_variance,
            "spectrum_decay": spectrum_decay
        }
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        inputs: Tensor,
        **kwargs
    ) -> Dict[str, NTKResult]:
        """Compare NTK spectra across multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            inputs: Common input coordinates
            **kwargs: Arguments passed to analyze_spectrum
            
        Returns:
            Dictionary of model_name -> NTKResult
        """
        results = {}
        
        for name, model in models.items():
            analyzer = NTKAnalyzer(model, device=self.device, dtype=self.dtype)
            results[name] = analyzer.analyze_spectrum(inputs, **kwargs)
            
        return results
    
    def print_spectrum_summary(self, result: NTKResult, top_k: int = 10):
        """Print a summary of spectrum analysis results."""
        print(f"\n=== NTK Spectrum Analysis ===")
        print(f"Model: {result.metadata['model_type']}")
        print(f"Samples: {result.metadata['num_samples']}")
        print(f"Parameters: {result.metadata['num_params']:,}")
        print(f"Input dimension: {result.metadata['input_dim']}")
        print(f"\n--- Spectrum Statistics ---")
        print(f"Trace: {result.trace:.4f}")
        print(f"Condition number: {result.condition_number:.2e}")
        print(f"Effective rank: {result.effective_rank:.2f}")
        print(f"Spectrum decay rate: {result.spectrum_decay:.4f}")
        
        # Top eigenvalues
        k = min(top_k, len(result.eigenvalues))
        print(f"\n--- Top {k} Eigenvalues ---")
        for i in range(k):
            evr = result.explained_variance[i] if i < len(result.explained_variance) else 0
            print(f"λ_{i+1:2d}: {result.eigenvalues[i]:8.4f} (EVR: {evr:.3f})")


def analyze_model_ntk(
    model: nn.Module,
    inputs: Tensor,
    *,
    device: Optional[torch.device] = None,
    top_k: int = 10,
    return_all_eigenvalues: bool = True,
    **kwargs
) -> NTKResult:
    """Convenience function for single-model NTK analysis.
    
    Args:
        model: Neural network model
        inputs: Input coordinates
        device: Computation device
        top_k: Number of top eigenvalues to return for convenience access
        return_all_eigenvalues: Whether to return all eigenvalues in the result
        **kwargs: Additional arguments for analyze_spectrum
        
    Returns:
        NTK analysis result
    """
    analyzer = NTKAnalyzer(model, device=device, default_top_k=top_k)
    return analyzer.analyze_spectrum(
        inputs, 
        top_k=top_k, 
        return_all_eigenvalues=return_all_eigenvalues,
        **kwargs
    )


def compare_model_ntks(
    models: Dict[str, nn.Module],
    inputs: Tensor,
    *,
    device: Optional[torch.device] = None,
    top_k: int = 10,
    **kwargs
) -> Dict[str, NTKResult]:
    """Convenience function for multi-model NTK comparison.
    
    Args:
        models: Dictionary of model_name -> model
        inputs: Input coordinates  
        device: Computation device
        top_k: Number of top eigenvalues to return
        **kwargs: Additional arguments for analyze_spectrum
        
    Returns:
        Dictionary of model_name -> NTKResult
    """
    if not models:
        raise ValueError("No models provided")
    
    # Use first model's device if not specified
    if device is None:
        device = next(iter(models.values())).parameters().__next__().device
    
    analyzer = NTKAnalyzer(list(models.values())[0], device=device, default_top_k=top_k)
    return analyzer.compare_models(models, inputs, top_k=top_k, **kwargs)


if __name__ == "__main__":
    # Basic test to ensure imports work
    print("NTK analysis module loaded successfully!")
    print("Available functions:")
    print("- NTKAnalyzer: Main analysis class")
    print("- analyze_model_ntk: Single model analysis")
    print("- compare_model_ntks: Multi-model comparison")
