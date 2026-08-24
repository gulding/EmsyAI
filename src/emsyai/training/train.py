import os
import math
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from emsyai.model.transformer import EmsyAIModel
from emsyai.tokenizer import BPETokenizer
from emsyai.training.dataset import prepare_dataset, get_dataloaders
from emsyai.training.evaluate import evaluate_perplexity, generate_sample

def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-5):
    """
    Cosine learning rate schedule with linear warmup.
    
    Why Warmup?
    At the start of training, the model weights are totally random. If we hit it with 
    a massive learning rate right away, the gradients will be huge, causing numerical 
    instability (or NaN losses). We "warm up" by linearly increasing the learning rate 
    over the first few hundred steps.
    
    Why Cosine Decay?
    As the model gets closer to the optimal solution, we want it to take smaller and 
    smaller steps so it doesn't overshoot the minimum. A cosine curve smoothly drops 
    the learning rate down to a minimum value.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
        
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    # --- Configuration ---
    # File paths
    text_path = "dataset/smollm_corpus_v2.txt"
    tokenizer_path = "dataset/tokenizer_v2.json"
    bin_path = "dataset/train_v2.bin"
    checkpoint_dir = "checkpoints_v2"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model Hyperparameters (EmsyAI-120M)
    vocab_size = 16000
    dim = 768
    n_layers = 12
    n_heads = 12
    n_kv_heads = 4
    hidden_dim = 2048
    seq_len = 1024
    
    # Training Hyperparameters
    micro_batch_size = 4
    gradient_accumulation_steps = 8
    # Effective batch size = 4 * 8 = 32
    max_steps = 5000
    warmup_steps = 200
    max_lr = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0
    checkpoint_interval = 500
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Enable TF32 for faster matmuls on Ampere+ GPUs (like RTX 3060)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # --- Data Preparation ---
    prepare_dataset(text_path, tokenizer_path, bin_path)
    train_loader, val_loader = get_dataloaders(bin_path, seq_len=seq_len, batch_size=micro_batch_size)
    train_iter = iter(train_loader)
    
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # --- Model & Optimizer Initialization ---
    model = EmsyAIModel(
        vocab_size=vocab_size, dim=dim, n_layers=n_layers, 
        n_heads=n_heads, n_kv_heads=n_kv_heads, 
        hidden_dim=hidden_dim, max_seq_len=seq_len
    )
    model.to(device)
    
    # Why Weight Decay?
    # It adds a small penalty to the size of the weights, preventing them from growing too large.
    # This prevents the model from relying too heavily on any single feature, acting as regularization.
    # We apply it to all 2D matrices but NOT to 1D biases or RMSNorm scales.
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95))
    
    # --- Mixed Precision Training ---
    # Why Mixed Precision?
    # Float32 uses 4 bytes per number. Bfloat16 uses 2 bytes.
    # We compute the forward/backward passes in Bfloat16 to double our speed and halve VRAM usage,
    # but we keep the master weights in Float32 so the optimizer updates are precise.
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    # --- Training Loop ---
    print(f"\nStarting training for {max_steps} steps...")
    model.train()
    optimizer.zero_grad()
    
    step = 0
    t0 = time.time()
    
    while step < max_steps:
        # Determine learning rate for this step
        lr = get_lr(step, max_steps, warmup_steps, max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Micro-batch gradient accumulation loop
        lossf = 0.0
        for micro_step in range(gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
                
            x, y = x.to(device), y.to(device)
            
            # Forward pass with mixed precision autocast
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                # Scale loss by accumulation steps so the final gradient is an average, not a sum
                loss = loss / gradient_accumulation_steps
                
            # Backward pass
            scaler.scale(loss).backward()
            lossf += loss.item()
            
        # Gradient Clipping
        # Why clip gradients? If a batch contains weird data, it might produce massive gradients
        # that explode the model weights and ruin training. We cap the norm of gradients to 1.0.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        step += 1
        
        # Logging
        if step % 10 == 0 or step == 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Tokens per second calculation
            tokens_processed = micro_batch_size * seq_len * gradient_accumulation_steps
            tok_sec = tokens_processed / dt
            
            if "cuda" in device:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                mem_mb = 0.0
                
            print(f"Step {step:4d} | Loss: {lossf:.4f} | LR: {lr:.2e} | Tok/sec: {tok_sec:.2f} | Mem: {mem_mb:.0f} MB")
            
        # Checkpointing & Evaluation
        if step % checkpoint_interval == 0:
            val_perp = evaluate_perplexity(model, val_loader, device)
            print(f"\n--- Validation at step {step} ---")
            print(f"Perplexity: {val_perp:.2f}")
            
            generate_sample(model, tokenizer, device)
            
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': lossf,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}\n")

if __name__ == "__main__":
    main()
