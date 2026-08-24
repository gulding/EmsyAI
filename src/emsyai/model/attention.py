import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from emsyai.model.embedding import apply_rotary_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads to match the number of Query heads in GQA.
    
    If we have 4 KV heads and 8 Q heads (n_rep = 2), this duplicates each KV head 
    so it can be paired with 2 Q heads during the dot product.
    
    Input shape: (Batch, SeqLen, KVHeads, HeadDim)
    Output shape: (Batch, SeqLen, KVHeads * n_rep, HeadDim)
    """
    if n_rep == 1:
        return x
    B, SeqLen, KVHeads, HeadDim = x.shape
    # Expand introduces a new dimension of size n_rep, which we then flatten into the heads dim
    x = x[:, :, :, None, :].expand(B, SeqLen, KVHeads, n_rep, HeadDim)
    return x.reshape(B, SeqLen, KVHeads * n_rep, HeadDim)

class GroupedQueryAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) and KV Cache.
    
    Why Grouped Query Attention?
    In standard Multi-Head Attention (MHA), we have an equal number of Q, K, and V heads.
    During text generation, we must cache all past K and V states in GPU memory (the KV Cache).
    For large models and long contexts, this KV Cache takes up massive amounts of VRAM.
    
    Multi-Query Attention (MQA) fixed this by using exactly 1 K and 1 V head for ALL Q heads.
    But this degraded model quality.
    
    GQA is the compromise: we use a small number of K and V heads (e.g., 4) shared across 
    a larger number of Q heads (e.g., 8). This saves ~50% of KV Cache memory while 
    maintaining almost the exact same reasoning quality as MHA.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        
        # Q, K, V projections
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        # Output projection
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # Scaling factor for Q*K dot product
        # Why scale by sqrt(d_k)? The dot product of two vectors of size d_k has a variance of d_k.
        # This large variance pushes the softmax into regions with extremely small gradients (vanishing gradients).
        # Scaling by 1/sqrt(d_k) keeps the variance at 1, ensuring stable softmax outputs.
        # We explicitly skip creating self.scale since F.scaled_dot_product_attention 
        # calculates the scaling factor natively and optimally in C++.

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        B, SeqLen, _ = x.shape
        
        # 1. Project input to Q, K, V
        xq = self.wq(x).view(B, SeqLen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, SeqLen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, SeqLen, self.n_kv_heads, self.head_dim)
        
        # 2. Apply Rotary Positional Embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 3. KV Cache update (for inference)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # Insert the new K and V states into the cache at the correct sequence position
            cache_k[:B, start_pos : start_pos + SeqLen] = xk
            cache_v[:B, start_pos : start_pos + SeqLen] = xv
            # For attention, we need all past keys and values up to the current position
            keys = cache_k[:B, : start_pos + SeqLen]
            values = cache_v[:B, : start_pos + SeqLen]
        else:
            # During training, we just use the current sequence
            keys, values = xk, xv
            
        # 4. Repeat KV heads to match Q heads (GQA magic happens here)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # 5. Transpose for matrix multiplication: (B, NumHeads, SeqLen, HeadDim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 6. Scaled Dot-Product Attention using PyTorch's native C++ SDPA kernel (FlashAttention-2)
        # This replaces manual Q*K^T and Softmax, saving memory and computing insanely fast.
        # It automatically dispatches to FlashAttention-2 or xFormers if available!
        is_causal = False
        if mask is None and SeqLen > 1:
            is_causal = True
            
        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask if not is_causal else None,
            dropout_p=0.0,
            is_causal=is_causal
        )
        
        # 7. Reshape and final linear projection
        # Transpose back to (B, SeqLen, NumHeads, HeadDim) and flatten heads
        output = output.transpose(1, 2).contiguous().view(B, SeqLen, -1)
        
        # Return output and the updated KV states (which we only return during inference)
        return self.wo(output), (xk, xv)
