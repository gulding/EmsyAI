import os
import torch
from emsyai.model.transformer import EmsyAIModel
from emsyai.training.dataset import get_dataloaders
from emsyai.training.evaluate import evaluate_perplexity

def run_tournament():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Checkpoint Tournament on {device}...")
    
    # 1. Setup Data Loader
    bin_path = "dataset/train_v3.bin"
    seq_len = 4096
    print(f"Loading validation dataset from {bin_path}...")
    _, val_loader = get_dataloaders(bin_path, seq_len=seq_len, batch_size=1)
    
    # 2. Initialize V3 Titan Model
    print("Initializing EmsyAI-v3 Titan Model architecture...")
    model = EmsyAIModel(
        vocab_size=16000, 
        dim=896, 
        n_layers=16, 
        n_heads=14, 
        n_kv_heads=2, 
        hidden_dim=2560,
        max_seq_len=seq_len,
        rope_theta=50000.0
    )
    model.to(device)
    
    # 3. Checkpoint Candidates
    checkpoints = [
        "checkpoints_v3/model_step_3500.pt",
        "checkpoints_v3/model_step_4000.pt",
        "checkpoints_v3/model_step_4500.pt",
        "checkpoints_v3/model_step_5000.pt",
    ]
    
    print("\n--- Starting Tournament (High-Precision: 100 Batches / 409,600 Tokens) ---")
    results = {}
    
    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"File not found: {ckpt_path}")
            continue
            
        print(f"Loading {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # High precision evaluation!
        perp = evaluate_perplexity(model, val_loader, device, max_batches=100)
        results[ckpt_path] = perp
        print(f"Result -> {ckpt_path}: Perplexity = {perp:.2f}\n")
        
    print("--- Tournament Results ---")
    if results:
        best_ckpt = min(results, key=results.get)
        for k, v in results.items():
            print(f"{k}: {v:.2f}")
        print(f"\n🏆 WINNER (Golden Base): {best_ckpt} 🏆")
    else:
        print("No checkpoints were evaluated.")

if __name__ == "__main__":
    run_tournament()
