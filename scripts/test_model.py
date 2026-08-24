import torch
from emsyai.model.transformer import EmsyAIModel

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_vram(model: torch.nn.Module, batch_size: int, seq_len: int) -> float:
    """Estimates VRAM in MB for training (weights + gradients + adam states + activations)"""
    params = count_parameters(model)
    # Weights (fp32) = 4 bytes
    # Gradients (fp32) = 4 bytes
    # Adam states (fp32) = 8 bytes
    # Total per parameter = 16 bytes
    model_mem = params * 16
    
    # Very rough activation memory estimation for Transformer (B * S * hidden_dim * layers * 4)
    activation_mem = batch_size * seq_len * model.dim * model.n_layers * 32 * 4 
    
    return (model_mem + activation_mem) / (1024 * 1024)

def main():
    print("Initializing EmsyAI Base Model (50M-80M specs)...")
    
    model = EmsyAIModel(
        vocab_size=8000,
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        hidden_dim=1408,
        max_seq_len=512
    )
    
    params = count_parameters(model)
    print(f"\nModel Stats:")
    print(f"Total Parameters: {params:,}")
    
    # Batch size 32, seq len 512
    vram_mb = estimate_vram(model, batch_size=32, seq_len=512)
    print(f"Estimated Training VRAM (bs=32, seq=512, fp32): {vram_mb:.2f} MB")
    print(f"Fits comfortably in RTX 3060 12GB? {'Yes' if vram_mb < 12000 else 'No'}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randint(0, 8000, (2, 512))
    logits, _ = model(dummy_input)
    print(f"Output shape (Batch, SeqLen, Vocab): {logits.shape}")
    assert logits.shape == (2, 512, 8000), "Shape mismatch!"
    
    print("Model architecture built and verified successfully.")

if __name__ == "__main__":
    main()
