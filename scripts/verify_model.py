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
    print("\n--- Verifying Causal Mask (Functional Test) ---")
    model = EmsyAIModel(vocab_size=100, dim=64, n_layers=1, n_heads=1, n_kv_heads=1, hidden_dim=128, max_seq_len=64)
    model.eval()
    
    # We test causality functionally: changing future tokens must NOT change past predictions.
    tokens_base = torch.tensor([[10, 20, 30, 40]])
    tokens_modified = torch.tensor([[10, 20, 30, 99]]) # Changed the last token
    
    with torch.no_grad():
        logits_base, _ = model(tokens_base)
        logits_modified, _ = model(tokens_modified)
        
    # The prediction at position 2 (which predicts the token after '30') 
    # should be mathematically identical in both runs, because it cannot "look ahead" at token 99.
    diff = torch.abs(logits_base[0, 2] - logits_modified[0, 2]).max().item()
    
    print(f"Max logit difference at position 2 (after altering position 3): {diff:.8f}")
    if diff == 0.0:
        print("✅ Causality Verified! Future tokens do not leak into the past.")
    else:
        print("❌ CAUSALITY LEAK DETECTED!")

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
