import os
import math
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np

from emsyai.model.transformer import EmsyAIModel
from emsyai.config import V4_CONFIG

def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-5):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
    args = parser.parse_args()

    # V4 Hyperparameters
    micro_batch_size = 1
    gradient_accumulation_steps = 32 # Effective batch = 32
    max_steps = 15000
    warmup_steps = 500
    max_lr = 3e-4
    min_lr = 1e-5
    weight_decay = 0.1
    grad_clip = 1.0
    checkpoint_interval = 500
    
    bin_path = "dataset/train_v4.bin"
    checkpoint_dir = "checkpoints_v4"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    print("Initializing EmsyAI-v4 Model (1024-dim, 16 layers, 16Q/4KV)...")
    model = EmsyAIModel(**V4_CONFIG.__dict__)
    model.to(device)
    
    # Weight Decay Groups
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95))
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    # Stateful Variables
    start_step = 0
    tokens_seen = 0
    
    # Checkpoint Resume Logic
    if args.resume:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Resuming from {latest_ckpt}...")
            
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            
            start_step = ckpt.get('step', 0)
            tokens_seen = ckpt.get('tokens_seen', 0)
            print(f"✓ Model & Optimizer restored. Fast-forwarding dataset to token {tokens_seen:,}...")
        else:
            print("No checkpoints found. Starting from scratch.")
            
    # Data Loader
    def data_generator(bin_path, seq_len, batch_size, start_token):
        if not os.path.exists(bin_path):
            print(f"Warning: {bin_path} not found. Returning dummy data for testing.")
            while True:
                yield torch.zeros((batch_size, seq_len), dtype=torch.long), torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        mmap = np.memmap(bin_path, dtype=np.uint16, mode='r')
        total_tokens = len(mmap)
        idx = start_token % total_tokens
        
        while True:
            if idx + (batch_size * seq_len + 1) > total_tokens:
                idx = 0
                
            batch_tokens = mmap[idx : idx + (batch_size * seq_len + 1)]
            batch_tokens = batch_tokens.astype(np.int64)
            
            x = torch.tensor(batch_tokens[:-1]).view(batch_size, seq_len)
            y = torch.tensor(batch_tokens[1:]).view(batch_size, seq_len)
            
            yield x, y
            idx += batch_size * seq_len

    train_iter = data_generator(bin_path, V4_CONFIG.max_seq_len, micro_batch_size, tokens_seen)
    
    print(f"\nStarting training from step {start_step} to {max_steps}...")
    model.train()
    optimizer.zero_grad()
    
    t0 = time.time()
    
    try:
        for step in range(start_step, max_steps):
            lr = get_lr(step, max_steps, warmup_steps, max_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            lossf = 0.0
            for micro_step in range(gradient_accumulation_steps):
                x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / gradient_accumulation_steps
                    
                scaler.scale(loss).backward()
                lossf += loss.item()
                tokens_seen += micro_batch_size * V4_CONFIG.max_seq_len
                
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if (step + 1) % 10 == 0 or step == start_step:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                tok_sec = (micro_batch_size * V4_CONFIG.max_seq_len * gradient_accumulation_steps) / dt
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if "cuda" in device else 0
                print(f"Step {step+1:4d} | Loss: {lossf:.4f} | LR: {lr:.2e} | Tok/sec: {tok_sec:.2f} | Mem: {mem_mb:.0f} MB | Tokens: {tokens_seen:,}")
                
            if (step + 1) % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step+1}.pt")
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'tokens_seen': tokens_seen,
                    'loss': lossf,
                    'lr': lr
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}\n")
                
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Caught Ctrl+C! Saving emergency checkpoint...")
        ckpt_path = os.path.join(checkpoint_dir, "model_interrupted.pt")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'tokens_seen': tokens_seen,
            'loss': lossf,
            'lr': lr
        }, ckpt_path)
        print(f"Emergency checkpoint saved to {ckpt_path}. Safe to close.")

if __name__ == "__main__":
    main()
