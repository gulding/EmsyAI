import json
import torch
from datasets import load_dataset
from tqdm import tqdm

from emsyai.chat import load_model
from emsyai.model.lora import apply_lora

def generate_pytorch(model, tokenizer, prompt, device, max_tokens=256, temp=0.2):
    # Simplistic generation loop adapted for exact stop token
    input_ids = tokenizer.encode(prompt, allowed_special=set())
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated = []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            for _ in range(max_tokens):
                logits, _ = model(x)
                # Take last token
                logits = logits[:, -1, :]
                
                # Apply temperature
                if temp > 0:
                    logits = logits / temp
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                
                # Check for EOS or other stop conditions if necessary
                if next_token_id == 2: # <|eos|>
                    break
                    
                generated.append(next_token_id)
                x = torch.cat((x, next_token), dim=1)
                
                # Truncate context if it gets too long
                if x.size(1) > 2048:
                    x = x[:, -2048:]
                    
    # Decode only the newly generated tokens
    return tokenizer.decode(generated)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    # 1. Load Base + LoRA via pure PyTorch
    model, tokenizer = load_model("checkpoints_v4/model_step_15000.pt", "dataset/tokenizer_v2.json", device, "v4")
    
    print("Applying LoRA adapters...")
    apply_lora(model, rank=8, alpha=16.0)
    lora_ckpt = torch.load("checkpoints_v4/lora/instruct_lora_step_10000.pt", map_location=device, weights_only=True)
    model.load_state_dict(lora_ckpt, strict=False)
    model.to(device)  # Move the newly injected LoRA layers to CUDA
    model.eval()
    
    print("Fetching HumanEval dataset...")
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    
    out_file = "humaneval_samples.jsonl"
    results = []
    
    print(f"Generating completions for {len(humaneval)} tasks (Pure PyTorch)...")
    for task in tqdm(humaneval):
        prompt = task["prompt"]
        task_id = task["task_id"]
        
        # Instruction format
        instruct_prompt = f"[USER]\nComplete the following Python code. Output ONLY the valid Python code for the function, nothing else.\n\n```python\n{prompt}```\n[MODEL]\n```python\n{prompt}"
        
        completion = generate_pytorch(model, tokenizer, instruct_prompt, device, max_tokens=256, temp=0.2)
        
        # Strip out markdown formatting if the model generated it
        if "```" in completion:
            completion = completion.split("```")[0]
            
        results.append({
            "task_id": task_id,
            "completion": completion
        })
        
        with open(out_file, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
                
    print(f"\nDone! Saved to {out_file}.")
    print("Run grading with: uv run evaluate_functional_correctness humaneval_samples.jsonl")

if __name__ == "__main__":
    main()
