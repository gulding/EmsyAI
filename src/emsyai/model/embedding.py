import torch
import torch.nn as nn
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the frequency tensor for complex exponentials (cis) used in RoPE.
    
    Formula:
    theta_i = 10000 ^ (-2(i-1)/d)
    m * theta_i = position * theta_i
    cis = cos(m * theta_i) + i * sin(m * theta_i)
    
    Why RoPE (Rotary Positional Embeddings)?
    Absolute positional embeddings (like in the original Transformer) add a fixed vector to the token embedding.
    This means the model struggles to understand relative distances (e.g., "token A is 3 steps away from token B").
    RoPE instead ROTATES the Query and Key vectors in high-dimensional space based on their absolute position.
    The dot product of two rotated vectors (Query * Key) naturally depends on their RELATIVE angle (distance),
    giving the model a perfect mathematical representation of relative distances.
    """
    # Create the frequencies for each dimension pair: [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Create the positions sequence: [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # Outer product to get all combinations of positions and frequencies
    # shape: (end, dim // 2)
    freqs = torch.outer(t, freqs)
    
    # Convert to complex numbers in polar form: cos(freqs) + i * sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the rotary embeddings to Queries (xq) and Keys (xk).
    
    Expected shapes:
    xq: (Batch, SeqLen, NumHeads, HeadDim)
    xk: (Batch, SeqLen, NumKVHeads, HeadDim)
    freqs_cis: (SeqLen, HeadDim // 2)
    """
    # Reshape xq and xk to treat adjacent elements as complex number pairs
    # shape: (B, SeqLen, NumHeads, HeadDim // 2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Convert pairs to complex numbers
    # shape: (B, SeqLen, NumHeads, HeadDim // 2)
    xq_complex = torch.view_as_complex(xq_)
    xk_complex = torch.view_as_complex(xk_)
    
    # Reshape freqs_cis for broadcasting: (1, SeqLen, 1, HeadDim // 2)
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    
    # Multiply complex numbers (this performs the rotation)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    
    # Cast back to original dtype (e.g., bfloat16)
    return xq_out.type_as(xq), xk_out.type_as(xk)
