import torch
from emsyai.config import V4_CONFIG
from emsyai.model.transformer import EmsyAIModel
import torch.nn.functional as F

print("Initializing V4 model with NanoGPT scaling...")
model = EmsyAIModel(**V4_CONFIG.__dict__)

# Test random forward pass
x = torch.randint(0, V4_CONFIG.vocab_size, (1, 1024))
y = torch.randint(0, V4_CONFIG.vocab_size, (1, 1024))

logits, _ = model(x)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
print(f"Initial Random Loss: {loss.item():.4f}")
print("Expected theoretical loss: ~9.6803 (ln(16000))")
