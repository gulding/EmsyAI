import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from emsyai.tokenizer import BPETokenizer
from emsyai.model.transformer import EmsyAIModel
from emsyai.config import V4_CONFIG
import math

def get_instruct_dataloader(tokenizer: BPETokenizer, batch_size: int = 4, seq_len: int = 2048):
    """
    Loads the instruct.jsonl file, formats the text, tokenizes it, and yields batches.
    """
    if not os.path.exists("dataset/instruct.jsonl"):
        raise FileNotFoundError("Run 'python -m emsyai.training.generate_instruct_data' first.")
        
    with open("dataset/instruct.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    # Format and tokenize
    tokenized_examples = []
    for item in data:
        prompt_text = f"[USER]\n{item['instruction']}\n[MODEL]\n"
        output_text = f"{item['output']}<|eos|>"
        
        prompt_tokens = tokenizer.encode(prompt_text, allowed_special=set())
        output_tokens = tokenizer.encode(output_text, allowed_special={"<|eos|>"})
        
        # Input tokens
        tokens = prompt_tokens + output_tokens
        
        # Labels: -100 for prompt tokens (so they don't contribute to loss), then the output tokens
        # We use -100 because PyTorch CrossEntropyLoss ignores it by default.
        labels = [-100] * len(prompt_tokens) + output_tokens
        
        tokenized_examples.append((tokens, labels))
        
    # Infinite generator yielding batches
    def batch_generator():
        while True:
            # Randomly sample 'batch_size' examples
            indices = torch.randint(0, len(tokenized_examples), (batch_size,))
            
            x_batch = torch.zeros((batch_size, seq_len), dtype=torch.long)
            y_batch = torch.full((batch_size, seq_len), -100, dtype=torch.long) # Default to -100 for padding
            
            for i, idx in enumerate(indices.tolist()):
                tokens, labels = tokenized_examples[idx]
                
                # Pad or truncate to seq_len + 1
                if len(tokens) > seq_len + 1:
                    tokens = tokens[:seq_len + 1]
                    labels = labels[:seq_len + 1]
                else:
                    # Pad tokens with 0, labels with -100
                    pad_len = seq_len + 1 - len(tokens)
                    tokens = tokens + [0] * pad_len
                    labels = labels + [-100] * pad_len
                    
                x_batch[i] = torch.tensor(tokens[:-1], dtype=torch.long)
                y_batch[i] = torch.tensor(labels[1:], dtype=torch.long)
                
            yield x_batch, y_batch
            
    return batch_generator()

def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-6):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def finetune():
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="Total SFT steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (must encompass the prompt + output)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate for full SFT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to V4 base model checkpoint")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting FULL PARAMETER Fine-Tuning on {device}...")
    
    tokenizer_path = "dataset/tokenizer_v2.json"
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Loading Base Model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    model = EmsyAIModel(**V4_CONFIG.__dict__)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Enable gradient checkpointing for VRAM savings during Full SFT
    # (Optional, but highly recommended for 1024-dim models during full backprop)
    # model.gradient_checkpointing_enable() # Can be added if standard HF structure is used
    
    model.to(device)
    model.train()
    
    # Setup Optimizer for ALL parameters
    weight_decay = 0.05
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    # Setup Dataloader
    print(f"Initializing Dataloader (Batch Size: {args.batch_size}, Seq Len: {args.seq_len})")
    dataloader = get_instruct_dataloader(tokenizer, batch_size=args.batch_size, seq_len=args.seq_len)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"\nStarting Training for {args.steps} steps...")
    
    pbar = tqdm(range(1, args.steps + 1))
    moving_loss = 0.0
    warmup_steps = min(500, args.steps // 10)
    
    out_dir = "checkpoints_v4/instruct"
    os.makedirs(out_dir, exist_ok=True)
    
    for step in pbar:
        # Cosine LR
        current_lr = get_lr(step, args.steps, warmup_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        step_loss = 0.0
        
        # Micro-batch accumulation
        for _ in range(args.grad_accum):
            x, y = next(dataloader)
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits, _ = model(x)
                B, T, V = logits.shape
                loss = loss_fn(logits.view(B*T, V), y.view(B*T))
                loss = loss / args.grad_accum
                
            scaler.scale(loss).backward()
            step_loss += loss.item()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        moving_loss = 0.9 * moving_loss + 0.1 * step_loss if moving_loss > 0 else step_loss
        pbar.set_description(f"Loss: {moving_loss:.4f} | LR: {current_lr:.2e}")
        
        # Save checkpoints
        if step % 1000 == 0 or step == args.steps:
            path = os.path.join(out_dir, f"instruct_model_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': moving_loss,
            }, path)
            print(f"\nSaved Full-Parameter SFT checkpoint to {path}")
            
    print(f"\nFull Parameter Fine-Tuning complete!")

if __name__ == "__main__":
    finetune()
