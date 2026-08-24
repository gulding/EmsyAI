import os
from huggingface_hub import HfApi

def push_readme():
    api = HfApi()
    repo_id = "gulding/EmsyAI"
    
    # Define the Model Card content
    readme_content = """---
license: mit
language:
- en
tags:
- from-scratch
- pytorch
- educational
- lora
- code-generation
---

# EmsyAI 🧠

EmsyAI is a **28 Million Parameter Decoder-only Transformer** built entirely from scratch in pure PyTorch for educational purposes. It is not a wrapper around the `transformers` library, but a fully hand-rolled implementation of a modern LLM architecture (heavily inspired by Llama 3).

## 🏗️ Model Architecture

- **Parameters**: ~28 Million
- **Layers**: 8
- **Hidden Dimension**: 512
- **Attention Heads**: 8 Query, 4 KV (Grouped Query Attention)
- **FFN Hidden Dimension**: 1408 (SwiGLU)
- **Context Length**: 512
- **Vocab Size**: 8000
- **Normalization**: Pre-RMSNorm
- **Positional Encoding**: Rotary Positional Embeddings (RoPE)

## 🔧 Training Details

### Phase 1: Base Pretraining
The base model was pretrained from scratch on ~40MB of Python source code (from the CPython repository) using a hand-rolled Byte-Pair Encoding (BPE) tokenizer. 
- **Optimizer**: AdamW with `bfloat16` mixed precision
- **Learning Rate**: Cosine decay with linear warmup
- **Final Perplexity**: ~10.77

### Phase 2: Instruction Fine-Tuning (LoRA)
The model was fine-tuned to follow instructions using a custom, from-scratch Low-Rank Adaptation (LoRA) implementation. 
- **Dataset**: `CodeAlpaca-20k`
- **LoRA Targets**: All linear layers (Q, K, V, O, w1, w2, w3)
- **Trainable Parameters**: 598,016 (2.11%)
- **Final Loss**: ~1.92

## 💻 Usage

To use this model, you will need the custom inference code from the GitHub repository:
[https://github.com/gulding/EmsyAI](https://github.com/gulding/EmsyAI)

Because the tokenizer and architecture are custom-built, this model is **not directly compatible** with the `transformers` `AutoModel` API. It is designed to be run using the `chat_instruct.py` script provided in the GitHub repo.

## 🎓 Educational Value
This project demonstrates how to build every layer of a modern Large Language Model from mathematical first principles, including the data pipeline, the Transformer core, LoRA adapters, and autoregressive KV-caching generation.
"""
    
    # Save it locally just in case
    with open("HF_README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print(f"Uploading README.md to {repo_id}...")
    api.upload_file(
        path_or_fileobj="HF_README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Successfully uploaded the Model Card to HuggingFace!")

if __name__ == "__main__":
    push_readme()
