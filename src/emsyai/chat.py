import argparse
import torch
import os
from emsyai.model.transformer import EmsyAIModel
from emsyai.tokenizer import BPETokenizer
from emsyai.model.generate import generate

def load_model(checkpoint_path: str, tokenizer_path: str, device: str):
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # We initialize the model with the exact same hyperparams used in training
    model = EmsyAIModel(
        vocab_size=8000, 
        dim=512, 
        n_layers=8, 
        n_heads=8, 
        n_kv_heads=4, 
        hidden_dim=1408,
        max_seq_len=512
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="EmsyAI Interactive Chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_step_5000.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="dataset/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) sampling")
    parser.add_argument("--rep_penalty", type=float, default=1.2, help="Repetition penalty (1.0 = none)")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found. Run training first.")
        return
        
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    
    print("\n" + "="*50)
    print(f"EmsyAI Chat REPL (Device: {device})")
    print(f"Settings: Temp={args.temperature}, Top-K={args.top_k}, Top-P={args.top_p}, RepPenalty={args.rep_penalty}")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input(">>> ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
                
            if not prompt.strip():
                continue
                
            print("\nGenerating...")
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
                device=device
            )
            
            print("-" * 50)
            print(output)
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Type 'quit' to exit.")
        except EOFError:
            break

if __name__ == "__main__":
    main()
