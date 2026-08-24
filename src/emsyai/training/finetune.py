import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from emsyai.tokenizer import BPETokenizer
from emsyai.model.transformer import EmsyAIModel
from emsyai.model.lora import apply_lora
from emsyai.chat import load_model

def get_instruct_dataloader(tokenizer: BPETokenizer, batch_size: int = 4, seq_len: int = 256):
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
        # We format as:
        # [USER]
        # Write a function...
        # [MODEL]
        # def function...<|eos|>
        text = f"[USER]\n{item['instruction']}\n[MODEL]\n{item['output']}<|eos|>"
        tokens = tokenizer.encode(text)
        tokenized_examples.append(tokens)
        
    # Infinite generator yielding batches
    def batch_generator():
        while True:
            # Randomly sample 'batch_size' examples
            indices = torch.randint(0, len(tokenized_examples), (batch_size,))
            
            x_batch = torch.zeros((batch_size, seq_len), dtype=torch.long)
            y_batch = torch.zeros((batch_size, seq_len), dtype=torch.long)
            
            for i, idx in enumerate(indices.tolist()):
                tokens = tokenized_examples[idx]
                # Pad or truncate to seq_len + 1
                if len(tokens) > seq_len + 1:
                    tokens = tokens[:seq_len + 1]
                else:
                    # Pad with 0 (which is <|pad|> or we can just use eos)
                    tokens = tokens + [0] * (seq_len + 1 - len(tokens))
                    
                x_batch[i] = torch.tensor(tokens[:-1], dtype=torch.long)
                y_batch[i] = torch.tensor(tokens[1:], dtype=torch.long)
                
            yield x_batch, y_batch
            
    return batch_generator()

def finetune():
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting LoRA Fine-tuning on {device}...")
    
    # 1. Load Base Model
    checkpoint_path = "checkpoints/model_step_5000.pt"
    tokenizer_path = "dataset/tokenizer.json"
    
    print("Loading Base Model...")
    model, tokenizer = load_model(checkpoint_path, tokenizer_path, device)
    
    # 2. Freeze Base Model unconditionally
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Apply LoRA (LoRALinear sets requires_grad=True on A and B matrices)
    apply_lora(model, rank=8, alpha=16.0)
    model.to(device)
    model.train()
    
    # 4. Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # 5. Setup Dataloader
    print(f"Initializing Dataloader (Batch Size: {args.batch_size}, Seq Len: {args.seq_len})")
    dataloader = get_instruct_dataloader(tokenizer, batch_size=args.batch_size, seq_len=args.seq_len)
    
    # 5. Training Loop
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"\nStarting Training for {args.steps} steps...")
    
    pbar = tqdm(range(1, args.steps + 1))
    moving_loss = 0.0
    
    for step in pbar:
        x, y = next(dataloader)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward with autocast for speed
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits, _ = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), y.view(B*T))
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        moving_loss = 0.9 * moving_loss + 0.1 * loss.item() if moving_loss > 0 else loss.item()
        pbar.set_description(f"Loss: {moving_loss:.4f}")
        
        # Save checkpoints periodically
        if step % 1000 == 0 or step == args.steps:
            os.makedirs("checkpoints/lora", exist_ok=True)
            path = f"checkpoints/lora/instruct_lora_step_{step}.pt"
            lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(lora_state_dict, path)
            print(f"\nSaved checkpoint to {path}")
            
    print(f"\nLoRA fine-tuning complete!")

if __name__ == "__main__":
    finetune()
