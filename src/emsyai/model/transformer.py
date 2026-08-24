import torch
import torch.nn as nn
from typing import Optional, Tuple
from emsyai.model.normalization import RMSNorm
from emsyai.model.attention import GroupedQueryAttention
from emsyai.model.feedforward import SwiGLUFeedForward
from emsyai.model.embedding import precompute_freqs_cis

class TransformerBlock(nn.Module):
    """
    A single Transformer layer (decoder-only).
    
    Structure:
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    
    Why Residual Connections (the `x + ...`)?
    In deep neural networks, gradients get multiplied repeatedly during backpropagation.
    If the values are < 1, the gradient shrinks to 0 (Vanishing Gradient).
    Residual connections act as an "express lane", allowing gradients to bypass the 
    transformation block unchanged. This allows us to train very deep networks.
    
    Why Pre-Norm?
    Originally, Transformers used Post-Norm: x = RMSNorm(x + Attention(x)).
    However, researchers found that applying normalization BEFORE the transformation (Pre-Norm)
    significantly improves training stability, as the residual pathway remains completely unmodified.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = SwiGLUFeedForward(dim, hidden_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # 1. Attention Block
        norm_x = self.attention_norm(x)
        attn_out, kv_states = self.attention(norm_x, start_pos, freqs_cis, mask, kv_cache)
        x = x + attn_out
        
        # 2. Feed Forward Block
        norm_x = self.ffn_norm(x)
        ffn_out = self.feed_forward(norm_x)
        x = x + ffn_out
        
        return x, kv_states

class EmsyAIModel(nn.Module):
    """
    The full EmsyAI Language Model.
    """
    def __init__(
        self, 
        vocab_size: int = 8000, 
        dim: int = 512, 
        n_layers: int = 8, 
        n_heads: int = 8, 
        n_kv_heads: int = 4, 
        hidden_dim: int = 1408,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token Embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final Normalization
        self.norm = RMSNorm(dim)
        
        # Output Head (Logits)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight Tying
        # Why Weight Tying? 
        # The embedding matrix translates token IDs to dense vectors.
        # The output head translates dense vectors back to token IDs.
        # These are effectively inverse operations. By forcing them to share the same weights, 
        # we save a massive amount of parameters (e.g., 4M parameters in this model) 
        # and act as a form of regularization.
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

    def forward(
        self, 
        tokens: torch.Tensor, 
        start_pos: int = 0,
        kv_caches: Optional[list] = None
    ) -> torch.Tensor:
        B, SeqLen = tokens.shape
        
        # Token Embedding
        h = self.tok_embeddings(tokens)
        
        # Slice the precomputed RoPE frequencies for the current sequence
        freqs_cis = self.freqs_cis[start_pos : start_pos + SeqLen].to(h.device)
        
        # Causal Mask Generation
        # We only need a mask during training (SeqLen > 1).
        # During single-token autoregressive generation, SeqLen == 1 and no mask is needed.
        mask = None
        if SeqLen > 1:
            # Create a matrix of -infinity above the diagonal, and 0 on/below the diagonal.
            # This ensures token i can only attend to tokens <= i (it can't see the future).
            mask = torch.full((SeqLen, SeqLen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            # Reshape for broadcasting with Attention scores: (B, NumHeads, SeqLen, SeqLen)
            # But wait, if start_pos > 0 (KV cache generation with chunks), mask needs to accommodate the cached tokens.
            # Usually mask is just (SeqLen, SeqLen + start_pos).
            mask = torch.hstack([torch.zeros((SeqLen, start_pos), device=h.device), mask])
            mask = mask.view(1, 1, SeqLen, SeqLen + start_pos)

        # Forward pass through all Transformer layers
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, updated_cache = layer(h, start_pos, freqs_cis, mask, layer_cache)
            if updated_cache is not None:
                new_kv_caches.append(updated_cache)
                
        # Final Norm and Projection
        h = self.norm(h)
        logits = self.output(h)
        
        return logits, new_kv_caches
