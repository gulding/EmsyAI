import torch
import json
import argparse
from gguf import GGUFWriter
import numpy as np
import os
from emsyai.model.transformer import EmsyAIModel

def load_and_merge(base_ckpt, lora_ckpt):
    device = "cpu"
    base_state = torch.load(base_ckpt, map_location=device, weights_only=True)
    lora_state = torch.load(lora_ckpt, map_location=device, weights_only=True)
    
    if "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]
        
    print(f"Loaded base model ({len(base_state)} tensors) and LoRA weights ({len(lora_state)} tensors)")
    
    merged_state = {}
    for k, v in base_state.items():
        merged_state[k] = v.clone()
        
    scaling = 16.0 / 8
    
    for k in list(lora_state.keys()):
        if "lora_A.weight" in k:
            prefix = k.replace("lora_A.weight", "")
            base_key = prefix + "linear.weight"
            
            A = lora_state[k]
            B = lora_state[prefix + "lora_B.weight"]
            
            update = (B @ A) * scaling
            
            if base_key in merged_state:
                merged_state[base_key] = merged_state[base_key] + update
            elif prefix + "weight" in merged_state:
                merged_state[prefix + "weight"] = merged_state[prefix + "weight"] + update
                
    return merged_state

def export_to_gguf(merged_state, tokenizer_path, out_file, version):
    print(f"Exporting {version} to GGUF...")
    writer = GGUFWriter(out_file, "llama")
    
    if version == "v3":
        writer.add_name("EmsyAI-v3-Titan-Instruct")
        writer.add_context_length(4096)
        writer.add_embedding_length(896)
        writer.add_block_count(16)
        writer.add_feed_forward_length(2560)
        writer.add_rope_dimension_count(64)
        writer.add_head_count(14)
        writer.add_head_count_kv(2)
        writer.add_layer_norm_rms_eps(1e-5)
        num_layers = 16
    else:
        writer.add_name("EmsyAI-v2-Instruct")
        writer.add_context_length(1024)
        writer.add_embedding_length(768)
        writer.add_block_count(12)
        writer.add_feed_forward_length(2048)
        writer.add_rope_dimension_count(64)
        writer.add_head_count(12)
        writer.add_head_count_kv(4)
        writer.add_layer_norm_rms_eps(1e-5)
        num_layers = 12
    
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
        
    from emsyai.tokenizer import BPETokenizer
    tok = BPETokenizer(vocab_size=16000)
    tok.load(tokenizer_path)
    
    writer.add_tokenizer_model("llama")
    
    tokens = []
    scores = []
    token_types = []
    
    for i in range(16000):
        if i < 256:
            tokens.append(bytes([i]))
        elif i in tok.inverse_special_tokens:
            tokens.append(tok.inverse_special_tokens[i].encode('utf-8'))
        elif i in tok.vocab:
            tokens.append(tok.vocab[i])
        else:
            tokens.append(f"<dummy_{i}>".encode('utf-8'))
            
        scores.append(float(i))
        
        if i in tok.inverse_special_tokens:
            token_types.append(3)
        else:
            token_types.append(1)
            
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(3)
    writer.add_pad_token_id(0)
    
    tensor_map = {
        "tok_embeddings.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "output.weight": "output.weight"
    }
    
    for i in range(num_layers):
        tensor_map[f"layers.{i}.attention.wq.weight"] = f"blk.{i}.attn_q.weight"
        tensor_map[f"layers.{i}.attention.wk.weight"] = f"blk.{i}.attn_k.weight"
        tensor_map[f"layers.{i}.attention.wv.weight"] = f"blk.{i}.attn_v.weight"
        tensor_map[f"layers.{i}.attention.wo.weight"] = f"blk.{i}.attn_output.weight"
        tensor_map[f"layers.{i}.feed_forward.w1.weight"] = f"blk.{i}.ffn_gate.weight"
        tensor_map[f"layers.{i}.feed_forward.w2.weight"] = f"blk.{i}.ffn_down.weight"
        tensor_map[f"layers.{i}.feed_forward.w3.weight"] = f"blk.{i}.ffn_up.weight"
        tensor_map[f"layers.{i}.attention_norm.weight"] = f"blk.{i}.attn_norm.weight"
        tensor_map[f"layers.{i}.ffn_norm.weight"] = f"blk.{i}.ffn_norm.weight"
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="checkpoints_v3/model_step_5000.pt")
    parser.add_argument("--lora", type=str, default="checkpoints_v3/lora/instruct_lora_step_10000.pt")
    parser.add_argument("--tokenizer", type=str, default="dataset/tokenizer_v2.json")
    parser.add_argument("--out", type=str, default="emsyai-v3-titan-instruct-f32.gguf")
    parser.add_argument("--version", type=str, default="v3")
    args = parser.parse_args()
    
    if not os.path.exists(args.base) or not os.path.exists(args.lora):
        print("Missing checkpoint files!")
        return
        
    merged = load_and_merge(args.base, args.lora)
    export_to_gguf(merged, args.tokenizer, args.out, args.version)
    
if __name__ == "__main__":
    main()
