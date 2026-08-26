import argparse
import torch
import os
from emsyai.model.transformer import EmsyAIModel
from emsyai.tokenizer import BPETokenizer
from emsyai.model.generate import generate
from emsyai.chat import load_model
from emsyai.model.lora import apply_lora

def main():
    parser = argparse.ArgumentParser(description="EmsyAI Instruct Chat (LoRA)")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints_v3/model_step_5000.pt")
    parser.add_argument("--lora_checkpoint", type=str, default="checkpoints_v3/lora/instruct_lora_step_10000.pt")
    parser.add_argument("--tokenizer", type=str, default="dataset/tokenizer_v2.json")
    parser.add_argument("--version", type=str, default="v3", choices=["v2", "v3"], help="Model version architecture")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--rep_penalty", type=float, default=1.2)
    parser.add_argument("--max_tokens", type=int, default=150)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Base Model
    print(f"Loading Base Model ({args.version.upper()})...")
    model, tokenizer = load_model(args.base_checkpoint, args.tokenizer, device, args.version)
    
    # 2. Inject LoRA Layers
    print("Injecting LoRA architecture...")
    apply_lora(model, rank=8, alpha=16.0)
    
    # 3. Load LoRA Weights
    print(f"Loading LoRA weights from {args.lora_checkpoint}...")
    lora_state = torch.load(args.lora_checkpoint, map_location=device, weights_only=True)
    # strict=False because the base model weights are already loaded, we only update the lora_A/B ones.
    model.load_state_dict(lora_state, strict=False)
    
    model.eval()
    model.to(device)
    
    print("\n" + "="*50)
    print(f"EmsyAI Instruct Chat (Device: {device})")
    print(f"Settings: Temp={args.temperature} (lowered for instruction following)")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("\n[USER]\n>>> ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
                
            if not prompt.strip():
                continue
                
            # Format the prompt using our exact training format
            formatted_prompt = f"[USER]\n{prompt}\n[MODEL]\n"
                
            print("\nGenerating...")
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
                device=device
            )
            
            print("-" * 50)
            # The generate function returns the prompt + output. We can slice it for cleaner reading.
            print(output)
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Type 'quit' to exit.")
        except EOFError:
            break

if __name__ == "__main__":
    main()
