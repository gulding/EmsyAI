import torch
import torch.nn.functional as F
import math
from emsyai.model.transformer import EmsyAIModel
from emsyai.model.generate import generate
from emsyai.tokenizer import BPETokenizer

@torch.no_grad()
def evaluate_perplexity(model: EmsyAIModel, val_loader, device: str, max_batches: int = 20) -> float:
    """
    Computes perplexity on the validation set.
    
    Why Perplexity?
    Cross-entropy loss outputs a single number, e.g., 2.5. But what does 2.5 mean?
    Perplexity is simply exp(Loss). If our loss is 2.5, exp(2.5) ≈ 12.18.
    This means intuitively: "When guessing the next token, the model is as confused 
    as if it were picking randomly from 12 equally likely options."
    
    A perfectly random model with an 8,000 vocab has a perplexity of 8,000.
    A good code model will drop to a perplexity of ~5 to 15.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
            
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        
        # Flatten logits and targets to compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        total_batches += 1
        
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    perplexity = math.exp(avg_loss)
    
    model.train() # Switch back to train mode
    return perplexity

def generate_sample(model: EmsyAIModel, tokenizer: BPETokenizer, device: str):
    """
    Generates a sample completion to see what the model is learning.
    """
    prompt = "def calculate_fibonacci(n):\n"
    print("\n" + "="*50)
    print("Generating Sample Completion...")
    print("="*50)
    
    try:
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            device=device
        )
        print(output)
    except Exception as e:
        print(f"Generation failed: {e}")
    print("="*50 + "\n")
