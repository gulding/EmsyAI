# EmsyAI V4 Pretraining Log

This log tracks the progression of the EmsyAI V4 (196M Parameter) model over its 15,000-step (2 Billion token) pretraining run on an RTX 3060 (12GB).

**Hardware Configuration:**
- Micro Batch Size: 1
- Gradient Accumulation: 32 (Effective Batch: 32)
- Sequence Length: 4096
- Peak VRAM Usage: ~8.0 GB
- Peak Compute: ~130W / 1890 MHz

> **Note:** Perplexity is calculated as `exp(Loss)`. A perfectly random model starts near ~18,000. A highly capable code model typically finishes between 5 and 15. True Tok/sec is calculated by multiplying the logged `Tok/sec` by 10 to compensate for the 10-step print logic.

| Step | Loss | Perplexity | Learning Rate | True Tok/sec | Total Tokens Seen |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 9.8593 | 19,135.4 | 6.00e-07 | ~7,583 | 131,072 |
| **500** | 3.8110 | 45.20 | 3.00e-04 | ~6,307 | 65,536,000 |
| **1000**| 2.9249 | 18.63 | 2.99e-04 | ~7,529 | 131,072,000 |
| **1500**| 2.6317 | 13.90 | 2.97e-04 | ~7,839 | 196,608,000 |
