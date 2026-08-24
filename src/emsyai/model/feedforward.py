import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FeedForward Network
    
    Formula:
    gate = Swish(x @ W_gate)   [where Swish/SiLU is x * sigmoid(x)]
    up = x @ W_up
    output = (gate * up) @ W_down
    
    Why SwiGLU?
    Traditional Transformers use a two-layer MLP with a ReLU or GELU activation: 
    Output = ReLU(x @ W1) @ W2.
    
    SwiGLU splits the first projection into two parallel projections (gate and up).
    It applies the Swish activation to the 'gate' and multiplies it element-wise with 'up'.
    This gating mechanism is highly expressive and allows the model to selectively route
    information. It has been empirically proven to yield better performance per parameter 
    than ReLU/GELU, which is why Llama, Qwen, and Mistral all use it.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # The gating projection (learns WHAT information to let through)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # The up projection (learns the actual transformed features)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # The down projection (brings the dimension back to model dim)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu is PyTorch's implementation of the Swish activation
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        # Element-wise multiplication, followed by down projection
        return self.w2(gate * up)
