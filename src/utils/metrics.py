import torch


@torch.jit.script
def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """JIT-compiled relative error computation."""
    with torch.no_grad():
        return torch.norm(pred - target) / (torch.norm(target) + 1e-8)

@torch.jit.script
def l2_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """JIT-compiled absolute error computation."""
    with torch.no_grad():
        return torch.norm(pred - target)