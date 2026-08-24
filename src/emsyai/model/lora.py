import math
import torch
import torch.nn as nn
from emsyai.model.transformer import EmsyAIModel

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper for a linear layer.
    
    Instead of training the original W matrix (which is frozen), we train two small 
    matrices A and B. The forward pass becomes:
    y = x @ W^T + x @ (A @ B)^T * (alpha / rank)
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        # Freeze the original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = linear.in_features
        out_features = linear.out_features
        
        # A matrix: (in_features -> rank)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        # B matrix: (rank -> out_features)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming (like standard linear)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # Initialize B with zeros so that the initial LoRA output is 0.
        # This means at step 0, the model acts EXACTLY like the original base model.
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base model forward
        base_output = self.linear(x)
        # LoRA forward
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_output + lora_output

def apply_lora(model: EmsyAIModel, rank: int = 8, alpha: float = 16.0) -> None:
    """
    Recursively replaces all Linear layers in the transformer with LoRALinear.
    Targeting all layers (Attention + FFN) gives the model much more capacity to learn.
    """
    for layer in model.layers:
        # Attention Projections
        layer.attention.wq = LoRALinear(layer.attention.wq, rank=rank, alpha=alpha)
        layer.attention.wk = LoRALinear(layer.attention.wk, rank=rank, alpha=alpha)
        layer.attention.wv = LoRALinear(layer.attention.wv, rank=rank, alpha=alpha)
        layer.attention.wo = LoRALinear(layer.attention.wo, rank=rank, alpha=alpha)
        
        # FFN Projections
        layer.feed_forward.w1 = LoRALinear(layer.feed_forward.w1, rank=rank, alpha=alpha)
        layer.feed_forward.w2 = LoRALinear(layer.feed_forward.w2, rank=rank, alpha=alpha)
        layer.feed_forward.w3 = LoRALinear(layer.feed_forward.w3, rank=rank, alpha=alpha)
        
    print(f"Applied LoRA to ALL linear layers (Rank={rank}, Alpha={alpha})")
    
    # Print trainable parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {trainable_params:,} / Total Params: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
