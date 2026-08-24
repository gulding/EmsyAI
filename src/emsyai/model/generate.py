import torch
import torch.nn.functional as F
from typing import List
from emsyai.model.transformer import EmsyAIModel
from emsyai.tokenizer import BPETokenizer

def init_kv_cache(model: EmsyAIModel, batch_size: int, device: torch.device):
    """
    Initializes empty KV Cache tensors for all layers.
    Shape: (Batch, MaxSeqLen, KVHeads, HeadDim)
    """
    cache = []
    head_dim = model.dim // (model.layers[0].attention.n_heads)
    n_kv_heads = model.layers[0].attention.n_kv_heads
    
    for _ in range(model.n_layers):
        k_cache = torch.zeros(
            batch_size, model.max_seq_len, n_kv_heads, head_dim, 
            device=device, dtype=torch.float32
        )
        v_cache = torch.zeros(
            batch_size, model.max_seq_len, n_kv_heads, head_dim, 
            device=device, dtype=torch.float32
        )
        cache.append((k_cache, v_cache))
    return cache

@torch.no_grad()
def generate(
    model: EmsyAIModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "cpu"
) -> str:
    """
    Autoregressive text generation using KV Cache and Repetition Penalty.
    """
    model.eval()
    model.to(device)
    
    # 1. Encode prompt
    input_ids = tokenizer.encode(prompt)
    seen_tokens = input_ids.copy()
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # 2. Initialize KV Cache
    kv_caches = init_kv_cache(model, batch_size=1, device=torch.device(device))
    
    # 3. Prefill phase (process the entire prompt at once to populate the cache)
    logits, _ = model(tokens, start_pos=0, kv_caches=kv_caches)
    next_token_logits = logits[:, -1, :]
    
    # Apply repetition penalty for the very first generated token
    if repetition_penalty != 1.0:
        unique_seen = list(set(seen_tokens))
        if unique_seen:
            scores = next_token_logits[0, unique_seen]
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            next_token_logits[0, unique_seen] = scores
    
    generated_ids = []
    start_pos = tokens.shape[1]
    
    next_token = _sample(next_token_logits, temperature, top_k, top_p)
    generated_ids.append(next_token.item())
    seen_tokens.append(next_token.item())
    
    # 4. Decode phase (autoregressively generate one token at a time)
    for _ in range(max_new_tokens - 1):
        tokens = next_token.unsqueeze(0) # Shape: (1, 1)
        
        logits, _ = model(tokens, start_pos=start_pos, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            unique_seen = list(set(seen_tokens))
            if unique_seen:
                scores = next_token_logits[0, unique_seen]
                scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
                next_token_logits[0, unique_seen] = scores
        
        next_token = _sample(next_token_logits, temperature, top_k, top_p)
        
        # Stop generating immediately if the model predicts <|eos|>
        if next_token.item() == tokenizer.special_tokens["<|eos|>"]:
            break
            
        generated_ids.append(next_token.item())
        seen_tokens.append(next_token.item())
        start_pos += 1
        
        # Stop if we hit max sequence length
        if start_pos >= model.max_seq_len:
            break
            
    return prompt + tokenizer.decode(generated_ids)

def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    Sampling logic for text generation.
    
    Why Temperature?
    Softmax outputs probabilities. If we divide the logits by a temperature < 1.0, the large 
    values get even larger relative to the small values, making the model more "confident" 
    and less random (greedy). If temp > 1.0, the probabilities flatten, increasing randomness.
    """
    if temperature <= 0.0:
        # Greedy decoding: always pick the most likely token
        return torch.argmax(logits, dim=-1)
        
    # Scale logits by temperature
    logits = logits / temperature
    
    # Top-K Sampling
    # We set the probabilities of all tokens except the top K to -infinity
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
    # Top-P (Nucleus) Sampling
    # We sort the probabilities, and keep adding tokens until their cumulative probability 
    # hits P. The rest are set to -infinity.
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter the mask back to the original ordering
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token.squeeze(1)
