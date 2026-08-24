---
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

EmsyAI is a complete, educational, decoder-only Transformer Language Model built entirely from scratch in pure PyTorch. The goal of this project is to demystify how modern Large Language Models (like Llama 3, Qwen, and GPT) work under the hood. 

This is not a wrapper around the `transformers` library. **Every single component was built from scratch:**
- Hand-rolled Byte-Pair Encoding (BPE) Tokenizer
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU Feed-Forward Networks
- RMSNorm Pre-Normalization
- KV Caching Autoregressive Inference
- Low-Rank Adaptation (LoRA) Fine-tuning

## 🏗️ Architecture

The architecture heavily mirrors modern LLM designs (specifically Llama 3):
- **Parameters**: ~89 Million (120M class with weight tying)
- **Layers**: 12
- **Hidden Dimension**: 768
- **Attention Heads**: 12 Query, 4 KV (GQA)
- **FFN Hidden Dimension**: 2048
- **Context Length**: 1024
- **Vocab Size**: 16000
- **Attention Kernel**: PyTorch Native FlashAttention-2 (SDPA)

## 🚀 Getting Started

This project uses `uv` for lightning-fast dependency management.

```bash
# Install dependencies
uv sync
```

### ⚡ Run Natively with Ollama
Because EmsyAI follows the standard Llama architecture, we exported it to GGUF format! You can run it instantly on your local machine using Ollama without installing any Python dependencies:

```bash
ollama run hf.co/gulding/EmsyAI
```

### 1. Tokenizer Training
The pure-Python BPE tokenizer is trained on the CPython source code.
```bash
# Download CPython source code
uv run python data/download_v2.py

# Train the BPE Tokenizer (creates dataset/tokenizer.json)
uv run python scripts/train_tokenizer.py
```

### 2. Base Model Pretraining
The training loop features AdamW, Cosine Learning Rate decay with linear warmup, Gradient Accumulation, and `bfloat16` Mixed Precision.

```bash
uv run python -m emsyai.training.train
```

### 3. LoRA Instruction Fine-Tuning
A custom LoRA implementation targets all linear layers (Q, K, V, O, w1, w2, w3) to fine-tune the base model on a subset of CodeAlpaca.

```bash
# Download 20k CodeAlpaca instruction pairs
uv run python -m emsyai.training.download_alpaca

# Train the LoRA adapters
uv run python -m emsyai.training.finetune --steps 10000
```

### 4. Exporting to GGUF
We include a custom `export_gguf.py` script that merges the base weights with the LoRA matrices and exports the full PyTorch model directly to a `.gguf` file compatible with `llama.cpp`.

```bash
uv run python scripts/export_gguf.py
```

### 5. Chat Interface
You can interact with the instruction-tuned model via a command-line REPL. It implements KV caching, top-k/top-p sampling, and repetition penalties.

```bash
uv run python -m emsyai.chat_instruct --lora_checkpoint checkpoints_v2/lora/instruct_lora_step_10000.pt
```

## 🛠️ Verification & Benchmarking
- Run `scripts/verify_model.py` to cryptographically prove that the Causal Mask prevents future token leakage, and that the KV Cache accurately matches standard autoregressive generation.
- Run `uv run python -m emsyai.benchmark` to test the base model's syntax generation capabilities using Python's AST parser.
