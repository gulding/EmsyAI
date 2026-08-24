import torch
import json
import argparse
from gguf import GGUFWriter
import numpy as np
import os
from emsyai.model.transformer import EmsyAIModel

def load_and_merge(base_ckpt, lora_ckpt):
    # Load base weights
    device = "cpu"
    base_state = torch.load(base_ckpt, map_location=device, weights_only=True)
    
    # Load LoRA weights
    lora_state = torch.load(lora_ckpt, map_location=device, weights_only=True)
    
    if "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]
        
    print(f"Loaded base model ({len(base_state)} tensors) and LoRA weights ({len(lora_state)} tensors)")
    
    # We will compute the merged weights natively
    merged_state = {}
    for k, v in base_state.items():
        merged_state[k] = v.clone()
        
    # LoRA parameters: rank 8, alpha 16.0 => scaling = 2.0
    scaling = 16.0 / 8
    
    # Find all base weight keys that have a corresponding LoRA A matrix
    # Our lora weights look like: layers.0.attention.wq.lora_A.weight
    for k in list(lora_state.keys()):
        if "lora_A.weight" in k:
            prefix = k.replace("lora_A.weight", "")
            base_key = prefix + "linear.weight" # In our model, LoRALinear wraps nn.Linear, so the base weight is linear.weight
            
            A = lora_state[k]
            B = lora_state[prefix + "lora_B.weight"]
            
            # W_new = W_old + (B @ A) * scaling
            update = (B @ A) * scaling
            
            if base_key in merged_state:
                merged_state[base_key] = merged_state[base_key] + update
            elif prefix + "weight" in merged_state:
                # In case it's named without 'linear.' in the base state
                merged_state[prefix + "weight"] = merged_state[prefix + "weight"] + update
                
    return merged_state

def export_to_gguf(merged_state, tokenizer_path, out_file):
    print("Exporting to GGUF...")
    writer = GGUFWriter(out_file, "llama")
    
    # Write Metadata
    writer.add_name("EmsyAI-120M-Instruct")
    writer.add_context_length(1024)
    writer.add_embedding_length(768)
    writer.add_block_count(12)
    writer.add_feed_forward_length(2048)
    writer.add_rope_dimension_count(64)
    writer.add_head_count(12)
    writer.add_head_count_kv(4)
    writer.add_layer_norm_rms_eps(1e-5)
    
    # Write Tokenizer
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
        
    vocab = tok_data["vocab"] if "vocab" in tok_data else None
    
    from emsyai.tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=16000)
    tok.load(tokenizer_path)
    
    tokens = []
    scores = []
    token_types = []
    
    for i in range(16000):
        # We need byte representation or string representation
        # Our decode returns a string (with replacement for invalid bytes)
        try:
            s = tok.decode([i])
            tokens.append(s.encode('utf-8'))
        except:
            tokens.append(f"<dummy_{i}>".encode('utf-8'))
        scores.append(0.0)
        
        if i in tok.inverse_special_tokens:
            token_types.append(3) # Control token
        else:
            token_types.append(1) # Normal
            
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(3)
    writer.add_pad_token_id(0)
    
    # Map Tensors
    tensor_map = {
        "tok_embeddings.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "output.weight": "output.weight"
    }
    
    for i in range(12):
        tensor_map[f"layers.{i}.attention.wq.weight"] = f"blk.{i}.attn_q.weight"
        tensor_map[f"layers.{i}.attention.wk.weight"] = f"blk.{i}.attn_k.weight"
        tensor_map[f"layers.{i}.attention.wv.weight"] = f"blk.{i}.attn_v.weight"
        tensor_map[f"layers.{i}.attention.wo.weight"] = f"blk.{i}.attn_output.weight"
        tensor_map[f"layers.{i}.feed_forward.w1.weight"] = f"blk.{i}.ffn_gate.weight"
        tensor_map[f"layers.{i}.feed_forward.w2.weight"] = f"blk.{i}.ffn_down.weight"
        tensor_map[f"layers.{i}.feed_forward.w3.weight"] = f"blk.{i}.ffn_up.weight"
        tensor_map[f"layers.{i}.attention_norm.weight"] = f"blk.{i}.attn_norm.weight"
        tensor_map[f"layers.{i}.ffn_norm.weight"] = f"blk.{i}.ffn_norm.weight"
        
        # If the linear layer was wrapped in LoRA, the base weight is named linear.weight
        tensor_map[f"layers.{i}.attention.wq.linear.weight"] = f"blk.{i}.attn_q.weight"
        tensor_map[f"layers.{i}.attention.wk.linear.weight"] = f"blk.{i}.attn_k.weight"
        tensor_map[f"layers.{i}.attention.wv.linear.weight"] = f"blk.{i}.attn_v.weight"
        tensor_map[f"layers.{i}.attention.wo.linear.weight"] = f"blk.{i}.attn_output.weight"
        tensor_map[f"layers.{i}.feed_forward.w1.linear.weight"] = f"blk.{i}.ffn_gate.weight"
        tensor_map[f"layers.{i}.feed_forward.w2.linear.weight"] = f"blk.{i}.ffn_down.weight"
        tensor_map[f"layers.{i}.feed_forward.w3.linear.weight"] = f"blk.{i}.ffn_up.weight"

    print("Converting and writing tensors...")
    for key, tensor in merged_state.items():
        if key in tensor_map:
            gguf_name = tensor_map[key]
            # Convert to numpy float32
            data = tensor.to(torch.float32).numpy()
            writer.add_tensor(gguf_name, data)
        else:
            if not key.endswith("lora_A.weight") and not key.endswith("lora_B.weight") and not "scaling" in key:
                print(f"Warning: Skipping unmapped tensor {key}")
            
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Success! GGUF model saved to {out_file}")

def main():
    base = "checkpoints_v2/model_step_5000.pt"
    lora = "checkpoints_v2/lora/instruct_lora_step_10000.pt"
    tokenizer = "dataset/tokenizer_v2.json"
    out = "emsyai-120m-instruct-f32.gguf"
    
    if not os.path.exists(base) or not os.path.exists(lora):
        print("Missing checkpoint files!")
        return
        
    merged = load_and_merge(base, lora)
    export_to_gguf(merged, tokenizer, out)
    
if __name__ == "__main__":
    main()
