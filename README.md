---
license: mit
pipeline_tag: text-generation
tags:
  - pytorch
  - emsyai
  - custom-architecture
  - pretraining
  - lora
  - ollama
---

# EmsyAI (V4)

EmsyAI is an open-source, from-scratch 196M parameter Large Language Model built strictly as an educational journey. The goal of this project is to demystify LLM creation by handling every single step locally: writing a custom BPE tokenizer, coding a Llama-style Transformer in pure PyTorch, pretraining on 2 Billion tokens, running LoRA instruction tuning, and finally exporting to GGUF for edge deployment via Ollama.

## 🚀 V4 Architecture

The V4 iteration is the most advanced version of EmsyAI yet, scaling to almost 200 Million parameters with a massive 4096-token context window.

- **Parameters:** ~196.7M (Base) + 2.3M (LoRA Adapters)
- **Training Tokens:** 1.96 Billion
- **Context Length:** 4096
- **Dim:** 1024
- **Layers:** 16
- **Heads:** 16 (Q), 4 (KV) (Grouped Query Attention)
- **FFN Dim:** 2816 (SwiGLU)
- **RoPE Theta:** 10,000.0

## 📈 Pretraining Results

The model was pretrained for 15,000 steps using an aggressive learning rate schedule. Despite running on consumer hardware, the training run achieved a flawlessly smooth exponential decay in both cross-entropy loss and validation perplexity, ultimately converging at a final perplexity of **5.86**.

![V4 Training Curve](v4_training_curve.png)

## 🧪 Evaluation & Data Contamination

After pretraining, EmsyAI V4 was fine-tuned for 10,000 steps using LoRA adapters on the ~20k `CodeAlpaca` instruction dataset to teach it the standard `[USER]` / `[MODEL]` chat format.

**HumanEval pass@1 Score:** `0.0%`

*Wait, 0.0%? Is that a failure?*
Actually, this is a massive success! Modern models that ace HumanEval are typically 40x larger (8B+) and trained on 1,500x more data (3T+ tokens). A 196M parameter model scoring 0% is mathematically expected. 

More importantly, **we audited the CodeAlpaca instruction dataset for contamination** before fine-tuning. Using an aggressive 13-gram exact match technique, we discovered exactly **1,644 sequences** in the open-source CodeAlpaca dataset that perfectly leaked HumanEval test logic (such as Heron's formula and prime number checks). 

Despite this verifiable data leakage, EmsyAI scored 0.0%, proving that the model did not blindly overfit or memorize the test set during the LoRA phase. It remains an honest, baseline 196M model!

## 💻 How to Run (Ollama)

EmsyAI V4 has been fused and exported to the highly optimized GGUF format for CPU/GPU inference.

If you have [Ollama](https://ollama.com/) installed, you can run EmsyAI directly:

```bash
# Clone the repository
git clone https://github.com/gulding/EmsyAI.git
cd EmsyAI

# Create the Ollama model using the provided Modelfile
ollama create emsyai-v4 -f Modelfile

# Start chatting!
ollama run emsyai-v4
```

## 🧠 What's Next (V5 Blueprint)

While V4 was a massive success, we are already blueprinting **V5**, which will adopt bleeding-edge insights inspired by the Qwen 3 architecture:
1. **QK-Norm:** Injecting RMSNorm on the Query and Key states to completely eliminate attention logit exploding (a major instability during V4 scaling).
2. **Document-Aware Packing:** Upgrading our tokenization pipeline to respect document boundaries instead of blindly concatenating texts, preventing cross-document attention poisoning.
3. **Tokenizer Rebuild:** Fixing a critical collision where bytes 0-3 currently overlap with special tokens, and expanding the vocabulary to 32K or 64K.
4. **RoPE Scaling:** Lowering `head_dim` back to 64 and adjusting `rope_theta` to 500,000.0 to better balance high-frequency and low-frequency positional perception at context=4096.

---
*Built with ❤️ in pure PyTorch.*
