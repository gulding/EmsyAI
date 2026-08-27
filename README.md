# EmsyAI 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch: 2.5.1+cu121](https://img.shields.io/badge/PyTorch-CUDA%2012.1-red.svg)](https://pytorch.org/)
[![Hugging Face: EmsyAI](https://img.shields.io/badge/HuggingFace-gulding%2FEmsyAI-orange.svg)](https://huggingface.co/gulding/EmsyAI)

EmsyAI is a complete, educational, decoder-only Transformer Language Model built entirely from mathematical first principles in pure PyTorch. The goal of this project is to demystify how modern LLMs (like Llama 3, Qwen 2.5, and DeepSeek) operate under the hood.

This is **not** a wrapper around the `transformers` library. **Every single component was built from scratch:**

- ✂️ **Hand-rolled Byte-Pair Encoding (BPE)** Tokenizer with native special token isolation (`<|eos|>`, `<|bos|>`, `<|pad|>`)
- 🔄 **Rotary Positional Embeddings (RoPE)** via complex polar tensor rotations
- ⚡ **Grouped Query Attention (GQA)** with fused FlashAttention-2 (PyTorch SDPA)
- 🚪 **SwiGLU Feed-Forward Networks** (3-matrix gated layout: $W_1, W_2, W_3$)
- ⚖️ **RMSNorm Pre-Normalization** with numerical float32 upcasting
- 💾 **Step-by-Step KV Caching** for high-throughput autoregressive inference
- 🎯 **Multi-Layer Low-Rank Adaptation (LoRA)** fine-tuning across all 7 linear projections

---

## 🏗️ Architecture Specifications

| Specification | EmsyAI-v3 Titan | EmsyAI-v4 (Current Training) |
|---|---|---|
| **Parameters** | 153.8 Million (Tied) | 196 Million (Tied) |
| **Layers ($L$)** | 16 | 16 |
| **Hidden Dimension ($d_{\text{model}}$)** | 896 | 1024 |
| **Attention Heads ($H_q / H_{kv}$)** | 14 Query / 2 KV (GQA 7:1) | 16 Query / 4 KV (GQA 4:1) |
| **FFN Dimension** | 2,560 (SwiGLU) | 2,816 (SwiGLU) |
| **Context Window ($T$)** | 4,096 tokens | 4,096 tokens |
| **Vocabulary Size** | 16,000 (Hybrid BPE) | 16,000 (Hybrid BPE) |
| **Attention Backend** | FlashAttention-2 | FlashAttention-2 (`is_causal=True`) |

---

## 🚀 Quickstart: Run in 5 Seconds via Ollama

You can run EmsyAI natively without installing PyTorch or GPU drivers:

```bash
ollama run hf.co/gulding/EmsyAI
```

## 💻 Developer & Training Pipeline

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# 1. Clone the repository
git clone https://github.com/gulding/EmsyAI.git
cd EmsyAI

# 2. Install dependencies (PyTorch with CUDA 12.1)
uv sync
```

### 1. Data Engine & Strict Decontamination
EmsyAI-v4 is trained on a strictly verified, mathematically sterile dataset. We explicitly filter out exact 13-gram overlap sequences against standard coding benchmarks (e.g., HumanEval) at the token level during ingestion.
```bash
# Stream and decontaminate 500M tokens from Cosmopedia v2
uv run python scripts/prepare_v4_data.py

# Verify the dataset is 100% sterile
uv run python scripts/audit_contamination.py
```

### 2. Pretraining
```bash
# Pretrain EmsyAI-v4 (196M Model @ 4,096 Context)
uv run python -m emsyai.training.train_v4
```

### 3. LoRA Instruction Fine-Tuning (SFT)
```bash
# Download 20k CodeAlpaca instruction dataset
uv run python -m emsyai.training.download_alpaca

# Train LoRA adapters across all linear layers
uv run python -m emsyai.training.finetune --steps 10000
```

### 4. Interactive Chat REPL
```bash
uv run python -m emsyai.chat \
  --checkpoint checkpoints_v4/model_step_15000.pt \
  --version v4
```

## 📝 Prompt Template

When querying the instruction model, use the exact role tags:

```text
[USER]
Write a Python function to check if a number is prime.
[MODEL]
```

## 🔬 Mathematical Verification Suite

Run our verification suite to mathematically validate attention causality and KV-cache consistency:

```bash
uv run python scripts/verify_model.py
```
