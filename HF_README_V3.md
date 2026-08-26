---
license: mit
pipeline_tag: text-generation
language:
- en
tags:
- gguf
- llama.cpp
- ollama
- from-scratch
- educational
- pytorch
---

# EmsyAI 🚀

EmsyAI is a small, decoder-only transformer language model built entirely from scratch in pure PyTorch — no `transformers` library, no pre-built blocks. It's an educational project: the goal is to demystify how models like Llama 3, Qwen 2.5, and DeepSeek work internally by actually building every piece — tokenizer, attention, RoPE, KV cache, LoRA — by hand.

Full training code, data pipeline, and developer docs live on [GitHub](https://github.com/gulding/EmsyAI). This page is about the model artifact itself: what it is, how to run it, and what to expect.

> [!WARNING]
> **Read this before you run it:** EmsyAI is a small model (V3 Titan is 154M parameters) trained on a small, narrow dataset. It is not a general-purpose assistant and won't behave like one — treat it as a working demonstration of a from-scratch LLM pipeline, not a production model. See [Limitations](#limitations).

### Built from scratch:
- 🧩 **Hand-rolled byte-pair encoding (BPE)** tokenizer
- 🔄 **Rotary position embeddings (RoPE)** via complex tensor rotation
- 🎯 **Grouped-query attention (GQA)**, fused via `F.scaled_dot_product_attention`
- 🧠 **SwiGLU** feed-forward layers
- 📏 **RMSNorm** pre-normalization
- 💾 **KV caching** for autoregressive inference
- 🎨 **LoRA fine-tuning** across all seven linear projections per block

## Quickstart

By default, we serve the latest **V3 Titan** architecture (154M parameters, 4,096-token context window).

```bash
ollama run hf.co/gulding/EmsyAI
```

or with `llama.cpp`:
```bash
llama-cli -hf gulding/EmsyAI:F32
```

> **Note:** If you are looking for the older **V2** architecture (88M params, 1,024-token context), the `emsyai-v2-instruct-f32.gguf` file is still available in the Files tab. 

## Prompt format

```text
[USER]
Write a Python function to check if a number is prime.
[MODEL]
```

## What's in this repo

| File | What it is | Do you need it? |
|---|---|---|
| `emsyai-v3-titan-instruct-f32.gguf` | **v3 Titan** base model + LoRA adapter, merged and exported to GGUF | **Yes.** This is what Ollama / llama.cpp actually run by default |
| `emsyai-v2-instruct-f32.gguf` | **v2** base model + LoRA adapter, merged and exported to GGUF | Optional. Legacy 88M architecture |
| `base_model_154M_step_5000.pt` | Raw PyTorch inference weights | Only if loading it back into the [training code](https://github.com/gulding/EmsyAI) |
| `tokenizer.json` | The trained 16k-vocab BPE tokenizer | Only needed for the PyTorch path, not GGUF |
| `Modelfile` | Ollama configuration — prompt template, stop token, temperature | Used automatically by `ollama run` |

*(GGUF export maps the model's tensors onto llama.cpp's naming convention for Ollama/llama.cpp compatibility — that's why the architecture shows as `llama` in the metadata above. The training-time implementation itself is fully custom, not derived from Llama's code.)*

## Model Details

| Specification | EmsyAI-v3 "Titan" (Latest) | EmsyAI-v2 (Legacy) |
|---|---|---|
| Parameters | 153.8M (tied embeddings) | 88.1M (tied embeddings) |
| Layers | 16 | 12 |
| Hidden dim | 896 | 768 |
| Attention heads | 14 Query / 2 KV (GQA 7:1) | 12 Query / 4 KV (GQA 3:1) |
| FFN dim | 2,560 (SwiGLU) | 2,048 (SwiGLU) |
| Context window | 4,096 tokens | 1,024 tokens |
| Vocab size | 16,000 (custom BPE) | 16,000 (custom BPE) |
| Position encoding | RoPE | RoPE |

**Training data (V3 Titan):** a ~150M-token hybrid of Cosmopedia v2 (synthetic educational text) and a Python code subset, tokenized with the project's own BPE tokenizer. Strict 10-gram decontamination was run against HumanEval and MBPP. Instruction-tuned with LoRA (rank 8, alpha 16) on ~20k CodeAlpaca examples with masked prompt loss.

## Project status

**V3 Titan is complete.** The model successfully converged during 5,000 steps of pretraining on the 150M-token corpus, followed by 10,000 steps of LoRA instruction tuning. Prompt loss masking (-100) successfully eliminated base-model "topic drift", resulting in a model that strictly outputs the desired python format. 

## Limitations

- **Small model, small data.** 154M active parameters trained on ~150M tokens — closer in scale to early GPT-2-class experiments than to a modern assistant model.
- **No RLHF or preference tuning** — only supervised LoRA fine-tuning on instruction-following examples. Expect repetitive, sometimes factually wrong, or off-topic output, especially past a few sentences.
- **Code it writes may not run.** This project's own benchmark only checks that generated Python parses, not that it's logically correct — don't assume otherwise.
- **English only**, and narrow in domain (educational text and Python code).

## License
MIT — see [LICENSE](https://github.com/gulding/EmsyAI/blob/main/LICENSE).

## Links
- 💻 [Source code, training pipeline, and full docs](https://github.com/gulding/EmsyAI)
- 💬 [Issues / discussions](https://github.com/gulding/EmsyAI/issues)
