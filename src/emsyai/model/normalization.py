import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm)
    
    Formula:
    RMSNorm(x) = (x / sqrt(Mean(x^2) + eps)) * weight
    
    Why RMSNorm?
    Traditional LayerNorm centers the activations (subtracts the mean) before scaling by the variance.
    Researchers found that the centering operation is computationally expensive and doesn't actually 
    help the model much. RMSNorm removes the mean-centering, scaling only by the Root Mean Square.
    This saves memory and compute (about 10-50% faster) while maintaining training stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The learnable scaling parameter (gamma), initialized to ones.
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the mean of the squares along the last dimension (the feature dimension).
        # keepdim=True ensures the shape stays (B, T, 1) so it broadcasts correctly when dividing.
        rms = torch.mean(x.pow(2), dim=-1, keepdim=True)
        # Scale the input by the inverse of the root mean square
        return x * torch.rsqrt(rms + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize and then multiply by the learnable weight parameter
        # We ensure x is cast to float32 before squaring to prevent overflow,
        # which is a common issue in mixed precision training.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
