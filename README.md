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
- **Parameters**: ~28 Million
- **Layers**: 8
- **Hidden Dimension**: 512
- **Attention Heads**: 8 Query, 4 KV (GQA)
- **FFN Hidden Dimension**: 1408
- **Context Length**: 512
- **Vocab Size**: 8000

## 🚀 Getting Started

This project uses `uv` for lightning-fast dependency management.

```bash
# Install dependencies
uv sync
```

### 1. Tokenizer Training
The pure-Python BPE tokenizer is trained on the CPython source code.
```bash
# Download CPython source code
uv run python data/download.py

# Train the BPE Tokenizer (creates dataset/tokenizer.json)
uv run python train_tokenizer.py
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

### 4. Chat Interface
You can interact with the instruction-tuned model via a command-line REPL. It implements KV caching, top-k/top-p sampling, and repetition penalties.

```bash
uv run python -m emsyai.chat_instruct --lora_checkpoint checkpoints/lora/instruct_lora_step_10000.pt
```

## 🛠️ Verification & Benchmarking
- Run `verify_model.py` to cryptographically prove that the Causal Mask prevents future token leakage, and that the KV Cache accurately matches standard autoregressive generation.
- Run `uv run python -m emsyai.benchmark` to test the base model's syntax generation capabilities using Python's AST parser.

## 🎓 Educational Value
If you are learning ML engineering, start by reading the code in `src/emsyai/model/`. The code is heavily commented to explain the math and the "why" behind each architectural decision.
