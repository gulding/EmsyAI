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

# EmsyAI 🧠

EmsyAI is a small, decoder-only transformer language model built entirely from scratch in pure PyTorch — no `transformers` library, no pre-built blocks. It's an educational project: the goal is to demystify how models like Llama 3, Qwen 2.5, and DeepSeek work internally by actually building every piece — tokenizer, attention, RoPE, KV cache, LoRA — by hand.

Full training code, data pipeline, and developer docs live on [GitHub](https://github.com/gulding/EmsyAI). This page is about the model artifact itself: what it is, how to run it, and what to expect.

> [!WARNING]
> **Read this before you run it:** EmsyAI is a small model (88M active parameters, hosted here) trained on a small, narrow dataset. It is not a general-purpose assistant and won't behave like one — treat it as a working demonstration of a from-scratch LLM pipeline, not a production model. See [Limitations](#limitations).

### Built from scratch:
- ✂️ **Hand-rolled byte-pair encoding (BPE)** tokenizer
- 🔄 **Rotary position embeddings (RoPE)** via complex tensor rotation
- ⚡ **Grouped-query attention (GQA)**, fused via `F.scaled_dot_product_attention`
- 🚪 **SwiGLU** feed-forward layers
- ⚖️ **RMSNorm** pre-normalization
- 💾 **KV caching** for autoregressive inference
- 🎯 **LoRA fine-tuning** across all seven linear projections per block

## Quickstart

```bash
ollama run hf.co/gulding/EmsyAI
```

or with `llama.cpp`:
```bash
llama-cli -hf gulding/EmsyAI:F32
```

> **Note:** This currently pulls EmsyAI-v2 (88M params, 1,024-token context, LoRA-instruction-tuned). EmsyAI-v3 "Titan" (178M params, 4,096-token context) is still training and will replace it here once training and instruction-tuning are finished — see [Project Status](#project-status) for where that stands right now.

## Prompt format

```text
[USER]
Write a Python function to check if a number is prime.
[MODEL]
```

## What's in this repo

| File | What it is | Do you need it? |
|---|---|---|
| `emsyai-v2-instruct-f32.gguf` | v2 base model + LoRA adapter, merged and exported to GGUF | **Yes.** This is what Ollama / llama.cpp actually run |
| `base_model_120M_step_5000.pt` | Raw PyTorch inference weights | Only if loading it back into the [training code](https://github.com/gulding/EmsyAI) |
| `tokenizer.json` | The trained 16k-vocab BPE tokenizer | Only needed for the PyTorch path, not GGUF |
| `Modelfile` | Ollama configuration — prompt template, stop token, temperature | Used automatically by `ollama run` |

*(GGUF export maps the model's tensors onto llama.cpp's naming convention for Ollama/llama.cpp compatibility — that's why the architecture shows as `llama` in the metadata above. The training-time implementation itself is fully custom, not derived from Llama's code.)*

## Model Details

| Specification | EmsyAI-v2 (hosted here) | EmsyAI-v3 "Titan" (training, not yet published) |
|---|---|---|
| Parameters | 88.1M (tied embeddings) | 153.8M (tied embeddings) |
| Layers | 12 | 16 |
| Hidden dim | 768 | 896 |
| Attention heads | 12 Query / 4 KV (GQA 3:1) | 14 Query / 2 KV (GQA 7:1) |
| FFN dim | 2,048 (SwiGLU) | 2,560 (SwiGLU) |
| Context window | 1,024 tokens | 4,096 tokens |
| Vocab size | 16,000 (custom BPE) | 16,000 (custom BPE) |
| Position encoding | RoPE | RoPE |

**Training data (v2):** a ~100k-sample hybrid of Cosmopedia v2 (synthetic educational text) and a Python code subset, tokenized with the project's own BPE tokenizer. Instruction-tuned with LoRA (rank 8, alpha 16) on ~20k CodeAlpaca examples.

## Project status

v3 "Titan" is training now, on a ~150M-token decontaminated corpus (checked against HumanEval/MBPP so benchmark problems aren't memorized). In the interest of being upfront about where a from-scratch model actually is mid-training, here's the real trajectory so far — perplexity is measured against a 16,000-word vocabulary, so ~16,000 is what random guessing would score:

| Step | Validation perplexity | What the sample completions look like |
|---|---|---|
| 500 | 36,987 | Empty — too early to produce anything |
| 1,000 | 780 | Empty — still too early |
| 1,500 | 485 | Fluent-sounding prose, not yet coherent |
| 2,000 | 272 | Python-shaped syntax, not yet logically correct |

For example, at step 2,000, prompting with `def calculate_fibonacci(n):` produced:
```python
for j in range(nums) return -1[i] # Return the total += list2.56, 9:4 = 0 for i in arr: x // 2 == 1
```
Recognizably Python-flavored — indentation, keywords, a comment — but not yet logically sound. That's the expected order for a small model early in training: surface fluency and syntax arrive before semantic correctness. This card will be updated with the v3 GGUF and real samples once training and instruction-tuning are complete.

## Limitations

- **Small model, small data.** 88.1M active parameters trained on ~100k documents — closer in scale to early GPT-2-class experiments than to a modern assistant model.
- **Short context.** 1,024 tokens for the hosted v2 model.
- **No RLHF or preference tuning** — only supervised LoRA fine-tuning on instruction-following examples. Expect repetitive, sometimes factually wrong, or off-topic output, especially past a few sentences.
- **Code it writes may not run.** This project's own benchmark only checks that generated Python parses, not that it's logically correct — don't assume otherwise.
- **English only**, and narrow in domain (educational text and Python code).

## License
MIT — see [LICENSE](https://github.com/gulding/EmsyAI/blob/main/LICENSE).

## Links
- 💻 [Source code, training pipeline, and full docs](https://github.com/gulding/EmsyAI)
- 🐛 [Issues / discussions](https://github.com/gulding/EmsyAI/issues)
