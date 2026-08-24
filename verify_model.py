import torch
import torch.nn.functional as F
from emsyai.model.transformer import EmsyAIModel
from emsyai.model.generate import init_kv_cache

def verify_kv_cache():
    print("--- Verifying KV Cache ---")
    model = EmsyAIModel(vocab_size=100, dim=64, n_layers=1, n_heads=4, n_kv_heads=2, hidden_dim=128, max_seq_len=64)
    model.eval()
    
    # Initialize cache
    kv_caches = init_kv_cache(model, batch_size=1, device=torch.device('cpu'))
    
    # Prefill 5 tokens
    tokens = torch.randint(0, 100, (1, 5))
    _, _ = model(tokens, start_pos=0, kv_caches=kv_caches)
    
    # Check cache shape for the single layer
    k_cache, v_cache = kv_caches[0]
    print(f"Initial Cache allocation shape (max_seq_len): {k_cache.shape}")
    
    start_pos = 5
    for i in range(10):
        # Generate 1 token at a time
        token = torch.randint(0, 100, (1, 1))
        _, _ = model(token, start_pos=start_pos, kv_caches=kv_caches)
        start_pos += 1
        
        # During the forward pass, we slice: cache_k[:B, : start_pos]
        # Let's print what the effective cached sequence length is
        print(f"Generated token {i+1} -> Effective KV Cache sequence length: {start_pos}")

def verify_causal_mask():
    print("\n--- Verifying Causal Mask ---")
    model = EmsyAIModel(vocab_size=100, dim=64, n_layers=1, n_heads=1, n_kv_heads=1, hidden_dim=128, max_seq_len=64)
    model.eval()
    
    # We will hook into the softmax probabilities
    attn_probs = []
    
    def hook_fn(module, input, output):
        # The output of F.softmax inside attention. We don't have a direct hook for intermediate locals,
        # but we can monkey-patch or just inspect the weights manually.
        pass
        
    # Since we can't easily hook a local variable, let's just trace or do a manual patch for the test
    import emsyai.model.attention as attention_mod
    original_softmax = F.softmax
    
    def mock_softmax(x, dim=-1):
        probs = original_softmax(x, dim)
        attn_probs.append(probs.detach())
        return probs
        
    attention_mod.F.softmax = mock_softmax
    
    tokens = torch.randint(0, 100, (1, 4))
    _ = model(tokens)
    
    attention_mod.F.softmax = original_softmax
    
    # Print the attention weights (B, NumHeads, SeqLen, SeqLen)
    weights = attn_probs[0][0, 0]
    print("Attention Weights Matrix (4x4 sequence):")
    print(weights.tolist())

def verify_overfit():
    print("\n--- Overfitting Sanity Test (Gradient Check) ---")
    vocab_size = 1000
    model = EmsyAIModel(
        vocab_size=vocab_size, dim=128, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=384, max_seq_len=64
    )
    model.train()
    
    # Tiny batch: 1 sequence of 32 tokens
    x = torch.randint(0, vocab_size, (1, 32))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting overfitting loop...")
    for i in range(101):
        logits, _ = model(x)
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = x[:, 1:].contiguous().view(-1)
        
        loss = F.cross_entropy(shift_logits, shift_labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 20 == 0:
            print(f"Step {i}: loss={loss.item():.4f}")

if __name__ == "__main__":
    verify_kv_cache()
    verify_causal_mask()
    verify_overfit()
